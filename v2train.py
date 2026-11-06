import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


df = pd.read_csv('quantum_training_data_master.csv').sample(n = 2500, random_state = 67)
features = torch.tensor(df["f1", "f2", "f3", "f4"], dtype=torch.float32)
labels = torch.tensor(df["steering"], dtype=torch.float32).view(-1,1)
