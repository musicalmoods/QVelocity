# QVelocity

> Autonomous Driving via Hybrid Quantum-Classical Machine Learning

Developed for the **IBM Qiskit Fall Fest 2026**, QVelocity is a fully autonomous driving pipeline that successfully pilots a vehicle through the Udacity Simulator.

Rather than relying purely on classical deep learning, QVelocity fuses PyTorch with Qiskit to create a **hybrid quantum-classical neural network**. The network calculates complex, continuous steering angles using only 16 trainable parameters by mapping the visual images and telemetry data to a quantum state.

---

## System Architecture
The driving pipeline operates in two distinct phases per frame: classical feature extraction and quantum expectation measurement.

* **Classical Vision:** Raw RGB camera feeds from the simulator are ingested and compressed by a modified ResNet18 model down to 4 features `f1, f2, f3, f4`
* **Quantum Angle Encoding** `(ZZFeatureMap)`: The 4 classical features are translated into physical rotations on 4 qubits. We utilize a Z-axis feature map to entangle the data for the model to analyze.
* **The Quantum Brain** `(RealAmplitudes)`: Serving as the actual "neural network," this hardware-efficient ansatz consists of 3 repeating layers of rotation and CNOT gates. It requires only 16 trainable parameters to learn the driving logic.
* **Continuous Steering** `(EstimatorQNN)`: Unlike classifiers that jerk the wheel hard left or right, we measure the Z-axis expectation value of the collapsed wave function to output a smooth, continuous steering decimal between -1.0 and 1.0.

---

## 🚀 Getting Started

### Prerequisites

* [Python 3.11](https://www.python.org/downloads/)
* [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim)

### 1. Installation

We strongly recommend running QVelocity in an isolated virtual environment to prevent dependency conflicts with other programs.

#### Clone the repository
```bash
git clone https://github.com/musicalmoods/QVelocity.git
cd QVelocity
```
#### Create and activate a virtual environment (Python 3.11 recommended)
```bash
python3.11 -m venv venv
```
#### Windows Activation:
```bash
venv\Scripts\Activate.ps1
```
#### Mac/Linux Activation:
```bash
source venv/bin/activate
```

#### Install the dependencies
```bash
pip install -r requirements.txt
```

### 2. Hitting the Track

#### To deploy the trained model to the simulator:

1. Run the server:
```bash
python drive.py
```

2. Open the **Udacity Simulator** and select **Autonomous Mode**. The car will instantly begin driving itself using the quantum network through the WebSocket connection.

## ⚠️ Troubleshooting: Intel 10th Gen+ CPU Crash
If you are running an Intel processor newer than 9th Generation (10th Gen Intel or later), the Udacity Simulator may instantly crash on startup due to a known OpenSSL compatibility bug in older versions of the Unity Engine.

To fix this, do not launch the simulator executable directly. Instead, create a `.bat` file in your simulator directory to set the correct environment variable bypass before launching.

1. Create a file named `gameplay.txt`
2. Add the following code:
```bat
@echo off
set OPENSSL_ia32cap=~0x200000200000000
start "" "beta_simulator.exe"
```
3. Save the file as `gameplay.bat`
4. Run `gameplay.bat` to open the simulator safely.

(Note: This environment variable change is strictly temporary. Because it is executed locally inside the `.bat` file, the `OPENSSL_ia32cap` variable only applies to this specific launch of the simulator. Your system's global environment variables remain untouched and revert back to normal as soon as the script finishes execution.)
