import base64
from io import BytesIO
from PIL import Image
import torch
import torchvision.models as models
from torchvision import transforms
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
import sys

# Fallback option
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vision_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
vision_model.fc = torch.nn.Linear(vision_model.fc.in_features, 4)
vision_model.eval().to(device)
vision_model.load_state_dict(torch.load('track2vision_weights.pt', weights_only=True)) # track2vision_weights for Dell model

num_qubits = 4
feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=1)
ansatz = RealAmplitudes(num_qubits=num_qubits, reps=3)
qc = feature_map.compose(ansatz)
observable = SparsePauliOp.from_list([("Z" * num_qubits, 1)])

qnn = EstimatorQNN(
    circuit=qc,
    observables=observable,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters
)
qnn_layer = TorchConnector(qnn)

# Security precautions
qnn_layer.load_state_dict(torch.load('qnn_weights_final.pt', map_location=device, weights_only=True))
qnn_layer.eval().to(device)

transform = transforms.Compose([
    transforms.Lambda(lambda img: img.crop((0, 60, 320, 140))),
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Boot up server
sio = socketio.Server(cors_allowed_origins='*')
app = Flask(__name__)

@sio.on('connect')
def connect(sid, environ):
    print(f"[SECURITY] Client connected with Session ID: {sid}. Starting engine...")
    sio.emit('steer', data={'steering_angle': '0.0', 'throttle': '0.0'})

@sio.on('disconnect')
def disconnect(sid):
    print(f"[SECURITY] Client {sid} disconnected. Engine stopped.")

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        try:
            if len(data.get("image", "")) > 500000:
                print("[SECURITY WARNING] Dropped abnormally large image payload.")
                return

            speed = float(data.get("speed", 0.0))
            image_data = base64.b64decode(data["image"])
            image = Image.open(BytesIO(image_data))
            
            img_tensor = transform(image).unsqueeze(0).to(device)
            steering_multi = 1
            with torch.no_grad():
                features = vision_model(img_tensor)
                steering_angle = qnn_layer(features).item() * steering_multi
            
            if abs(steering_angle) > 0.2 * steering_multi:
                throttle = 0
            elif speed < 15:
                throttle = 0.5
            else:
                throttle = 0.1
            
            sio.emit('steer', data={
                'steering_angle': str(steering_angle),
                'throttle': str(throttle)
            })
            
        except Exception as e:
            print(f"Server Error during telemetry parsing: {e}")
            sio.emit('manual', data={})
    else:
        sio.emit('manual', data={})

if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    
    server_socket = eventlet.listen(('localhost', 4567))
    print("[SYSTEM] Server bound to port 4567. Waiting for simulator...")
    
    try:
        eventlet.wsgi.server(server_socket, app)
        
    except KeyboardInterrupt:
        print("\n[SYSTEM] Shutdown signal received (Ctrl+C).")
        print("[SYSTEM] Severing socket connections and closing Port 4567...")
        server_socket.close()
        sys.exit(0)