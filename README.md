# QVelocity

> Autonomous Driving via Hybrid Quantum-Classical Machine Learning

Developed for the **IBM Qiskit Fall Fest 2026**, QVelocity is a fully autonomous driving pipeline that successfully pilots a vehicle through the Udacity Simulator.

Rather than relying purely on classical deep learning, QVelocity fuses PyTorch with Qiskit to create a **hybrid quantum-classical neural network**. The network calculates complex, continuous steering angles using only 16 trainable parameters by mapping the visual images and telemetry data to a quantum state.

---

## 🚀 Getting Started

### Prerequisites

* Python 3.11
* Udacity Self-Driving Car Simulator

### 1. Installation

We strongly recommend running QVelocity in an isolated virtual environment to prevent dependency conflicts.

# Clone the repository
```bash
git clone https://github.com/musicalmoods/Qiskit-Fall-Fest-2026.git
cd Qiskit-Fall-Fest-2026
```
# Create and activate a virtual environment (Python 3.11 recommended)
```bash
python3.11 -m venv venv
```
# Windows Activation:
```bash
venv\Scripts\Activate.ps1
```
# Mac/Linux Activation:
```bash
source venv/bin/activate
```

# Install the dependencies
```bash
pip install -r requirements.txt
```

### 2. Hitting the Track

To deploy the trained model to the simulator:

1. Run the server:
```bash
python drive.py
```

2. Open the **Udacity Simulator** and select **Autonomous Mode**. The car will instantly begin driving using the quantum network.
