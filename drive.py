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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vision_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
vision_model.fc = torch.nn.Linear(vision_model.fc.in_features, 4)
vision_model.eval().to(device)

# QC Brain Setup
num_qubits = 4
qc = ZZFeatureMap(feature_dimension=4, reps=1).compose(RealAmplitudes(num_qubits=4, reps=3))
observable = SparsePauliOp.from_list([("Z" * 4, 1)])
qnn = EstimatorQNN(circuit=qc, observables=observable)
qnn_layer = TorchConnector(qnn)
qnn_layer.load_state_dict(torch.load('qnn_weights.pt'))
qnn_layer.eval().to(device)

# Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.crop((0, 60, 320, 140))),
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Websocket connection to game engine
sio = socketio.Server()
app = Flask(__name__)

def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    }, skip_sid=True)

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = vision_model(img_tensor)
            steering_angle = qnn_layer(features).item()

        # Throttle if needed
        speed = float(data["speed"])
        throttle = 0.2 if speed < 12 else 0.0
        
        send_control(steering_angle, throttle)
    else:
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("Simulator Connected!")
    send_control(0, 0)

if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)