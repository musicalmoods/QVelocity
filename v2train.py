import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from tqdm import tqdm


# Extract the data first and set up the tensors
raw_df = pd.read_csv('track2quantum_training_data_master.csv')
straight_driving = raw_df[raw_df['steering'] == 0.0]
curve_driving = raw_df[raw_df['steering'] != 0.0]
straight_driving = straight_driving.sample(frac=0.2, random_state=42)
balanced_df = pd.concat([straight_driving, curve_driving]).sample(n=1200, random_state=67)
features = torch.tensor(balanced_df[["f1", "f2", "f3", "f4"]].values, dtype=torch.float32)
labels = torch.tensor(balanced_df["steering"].values, dtype=torch.float32).view(-1,1)

# Loading the data ()
dataset = TensorDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# Building the Quantum Circuit
qubits:int = 4
encoder = ZZFeatureMap(feature_dimension=qubits, reps=1)
ansatz = RealAmplitudes(num_qubits=qubits, reps=3)
circuit = encoder.compose(ansatz)
observable = SparsePauliOp.from_list([("Z" * qubits, 1)])


# Bridging PyTorch and QNN
qnn = EstimatorQNN(circuit=circuit, observables=observable,input_params=encoder.parameters,weight_params=ansatz.parameters)
model = TorchConnector(qnn)


# Optimizer and calculating Losses
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()


# Training the Quantum Model
epochs = 25
for epoch in range(epochs):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for i, (batch_features, batch_labels) in enumerate(progress_bar):
        optimizer.zero_grad() # Clean slate for every batch
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward() # Qiskit intercepts it and uses the Parameter-Shift Rule
        optimizer.step()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    print(f"Epoch: {epoch+1} Completed!")


torch.save(model.state_dict(), 'TRACK2DELL_weights.pt')
# We use state_dict instead of torch.save() so that if the model architecture changes, the mathematical numbers stay the
# same, so we can keep using the same model without it getting outdated.