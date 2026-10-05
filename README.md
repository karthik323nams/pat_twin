# PAT-Twin v0.4.0: Physics-Faithful FSOC Digital Twin & AI Benchmark Suite

PAT-Twin is a digital twin for Free-Space Optical Communications (FSOC) Pointing, Acquisition, and Tracking (PAT) systems. Unlike naive vision trackers that merely report pixel offsets, PAT-Twin links tracking accuracy directly to optical communication metrics (received power, pointing loss, link margin, and BER) and features an IMM-EKF multi-model estimator with an AI candidate spot verifier.

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Interactive GUI Dashboard
python -m pat.gui
# or:
python pat_gui.py

# 3. Run Benchmark Suite (Crosses all motions x weathers x noise)
python -m pat.bench --quick

# 4. CLI Headless Run with live optical telemetry
python -m pat.run --motion figure8 --weather clear --show

# 5. Export and Run Benchmark 2 Video Mode
python -m pat.run --export-video bench.mp4 --screen 1000 --duration 15 --motion random
python -m pat.run --video bench.mp4 --out logs/
```

---

## Key Capabilities

### 1. FSOC Optical Physics & Link Budget Engine (`pat/optics.py`)
* Converts live pointing error $\theta_e$ into Gaussian beam pointing loss $L_p = \exp(-2(\theta_e / w)^2)$.
* Computes instant received optical power ($P_{\text{rx}}$ in dBm and $\mu\text{W}$), atmospheric extinction across weather regimes (clear, haze, fog, rain, low-light), and link margin above receiver sensitivity ($-38\text{ dBm}$).
* Predicts dynamic Bit Error Rate (BER) and flags link status: **LOCKED** ($>3\text{ dB}$ margin), **MARGINAL** ($0-3\text{ dB}$ margin), or **OUTAGE** ($<0\text{ dB}$).

### 2. Hybrid IMM-EKF Multi-Model Tracker (`pat/track.py`)
* Runs three kinematic models concurrently:
  * **Model 0 (CV)**: Constant Velocity for smooth cruising and ballistic drift.
  * **Model 1 (CA)**: Constant Acceleration for thruster burns and speed changes.
  * **Model 2 (CT)**: Coordinated Turn for curved, figure-8, and spiral trajectories.
* Dynamically updates mode probabilities $[\mu_{\text{CV}}, \mu_{\text{CA}}, \mu_{\text{CT}}]$ based on measurement innovation likelihood.
* Adaptive measurement covariance $R$ scales with detected SNR and weather contrast.

### 3. AI-Assisted Candidate Verifier (`pat/ai_verifier.py`)
* Compact, CPU-optimized Convolutional Neural Network (ConvNet) scoring $32\times 32$ candidate patches.
* Discriminates true optical Gaussian/Airy spots from single-pixel sensor defects (hot pixels), background star clusters, cloud edges, and false decoy glints.
* Runs in pure vectorized NumPy with zero external dependency overhead ($<1\text{ ms}$ CPU inference time).

### 4. Dual-Stage Coarse-Fine Pointing Control
* **Coarse Stage**: Gimbal model with $5^\circ/\text{s}$ rate saturation, first-order lag, and deadbeat feedforward control.
* **Fine Stage**: Fast Steering Mirror (FSM) / electronic image stabilization (EIS) that cancels high-frequency vibration and jitter, maintaining optical alignment strictly $\le 10\text{ px}$.

### 5. Extended Target Motions & Multi-Target/Decoy Support (`pat/sim.py`)
* Motions: `straight`, `circular`, `figure8`, `random` (Ornstein-Uhlenbeck), `spiral`, `sinusoidal`, and `leo` (Keplerian orbital pass).
* Decoy generator: simulates false optical targets to verify tracker distractor rejection.

---

## CLI Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--motion` | `circular` | Target motion: `straight`, `circular`, `figure8`, `random`, `spiral`, `sinusoidal`, `leo` |
| `--weather` | `clear` | Atmosphere: `clear`, `haze`, `fog`, `rain`, `lowlight` |
| `--noise` | `sp,gauss` | Active noise: `sp`, `gauss`, `poisson` |
| `--sigma` | `15` | Gaussian noise standard deviation ($\le 20$) |
| `--jitter` | `8` | Camera jitter amplitude in $\pm\text{px/frame}$ |
| `--decoys` | `0` | Number of false decoy targets |
| `--no-imm` | `False` | Fallback to baseline single-model Kalman filter |
| `--no-ai` | `False` | Disable AI candidate verifier |
| `--no-fsm` | `False` | Disable Fine Steering Mirror stage |
| `--no-cue` | `False` | Disable wide-area acquisition sensor |
| `--show` | `False` | Render live OpenCV display window |
