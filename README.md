# 🛡️ THREAT CALL — AI-Powered Impersonation Defense & Cyber SOC System

> **Smart India Hackathon 2026** | Enterprise Cyber Security & Voice Deepfake Defense Platform

THREAT CALL is a real-time cybersecurity threat detection and impersonation defense system designed to intercept voice cloning, neural TTS deepfakes, retrieval-based voice conversion (RVC), video face-swapping, and social engineering vishing scams.

---

## 🌟 Key Features

1. **🛡️ Command Center & Global Live Threat Radar**: Animated HTML5 canvas radar sweep tracking live threat nodes and incoming vishing vectors.
2. **🎙️ Voice Forensics & Synthetic Sample Generator**: Multi-feature acoustic analysis (2D FFT, Pitch $F_0$ variance, MFCCs, Spectral Flatness) with built-in synthetic WAV playback (`/sample_audio`).
3. **👤 Voice Biometrics Verification Matrix**: Formant frequency resonance analysis ($F_1-F_4$), Pitch Jitter, Amplitude Shimmer, and Vocal Tract Length estimation against executive speaker footprints.
4. **👁️ Video Forensics (Face Shield)**: Real-time web camera frame spectrum analysis for video deepfake detection.
5. **🧬 Attack DNA Fingerprinting & Campaign Clustering**: Cryptographic attack signatures (`ATK-DNA-xxxx`) linking related threat calls to adversary campaigns (`CAMP-xxxx`).
6. **🔗 Deception Chain & Context-Aware Trust Engine**: 4-stage social engineering tracker (Authority Claim $\rightarrow$ Urgency $\rightarrow$ Isolation $\rightarrow$ Call to Action) with multi-factor Trust Score calculation.
7. **⚡ Adaptive Defense Engine & Action Firewall**: Tiered automated defense matrix and real-time interceptor blocking financial wire transfers and Remote Access Tools (*AnyDesk, TeamViewer, RustDesk*).
8. **📑 Official SOC Forensic Report Exporter**: Downloadable forensic audit reports with SHA-256 incident response checksums and **MITRE ATT&CK** taxonomy mappings (`T1566.004 Vishing`, `T1656 Impersonation`).

---

## 🛠️ Technology Stack

* **Backend**: Python 3.13, FastAPI, PyTorch, Librosa, NumPy, OpenCV, SQLite3, Uvicorn
* **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism Dark Theme), Javascript, Web Audio API, HTML5 Canvas
* **Database**: SQLite (`threat_call.db`)

---

## 🚀 Quick Start & Running Locally

1. **Clone the Repository**:
   ```bash
   git clone <your-repo-url>
   cd "project work 2"
   ```

2. **Install Dependencies**:
   ```bash
   pip install fastapi uvicorn torch librosa numpy opencv-python
   ```

3. **Run Backend Server**:
   ```bash
   python backend.py
   ```

4. **Access Dashboard**:
   Open browser at: `http://127.0.0.1:8000` or `http://localhost:8000`

---

## 📄 License
Developed for Smart India Hackathon (SIH) 2026.
