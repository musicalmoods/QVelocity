import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

df = pd.read_csv('quantum_training_data_master.csv').sample(n=1000, random_state=42)

features = torch.tensor(df[['f1', 'f2', 'f3', 'f4']].values, dtype=torch.float32)
labels = torch.tensor(df['steering'].values, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

qubits = 4
encoder = ZZFeatureMap(feature_dimension=qubits, reps=1)
ansatz = RealAmplitudes(num_qubits=qubits, reps=3)
circuit = encoder.compose(ansatz)
observable = SparsePauliOp.from_list([("Z" * qubits, 1)])

qnn = EstimatorQNN(
    circuit=circuit,
    observables=observable,
    input_params=encoder.parameters,
    weight_params=ansatz.parameters
)

model = TorchConnector(qnn)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

epochs = 5
print("Starting training loop")

for epoch in range(epochs):
    epoch_loss = 0.0
    
    for i, (batch_features, batch_labels) in enumerate(dataloader):
        print(f"Epoch {epoch} | Computing batch {i+1}/{len(dataloader)}...", end='\r')
        
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        print(f"Epoch {epoch} | Batch {i+1}/{len(dataloader)} | Loss: {loss.item():.4f}" + " " * 10)
        
    avg_loss = epoch_loss / len(dataloader)
    print(f"====> Epoch {epoch} Complete | Average Loss: {avg_loss:.4f}\n")

torch.save(model.state_dict(), 'qnn_weights.pt')
print("Training complete!")