<div align="center">
  <h1>🧅 Deceptive-Net</h1>
  <p><b>A Generative AI-Driven Financial Fraud Detection & Cyber Deception Framework</b></p>
</div>

<br>

Deceptive-Net is an advanced, academic-tier cybersecurity platform built to simultaneously detect anomalous financial transactions in real-time and actively deceive potential insider threats or attackers through data watermarking and simulated network topologies. 

## ✨ Key Features

* 🚀 **Real-Time Fraud Detection:** Analyzes streaming synthetic transactions using a robust machine learning ensemble.
* 🤖 **AI Ensemble Engine:** Combines LightGBM (Gradient Boosting) for high-speed classification with a PyTorch Autoencoder for unsupervised anomaly detection based on reconstruction error.
* 🛡️ **Cyber Deception (Honey Tokens):** Automatically injects traceable, watermarked transaction data (`HTXN`) into CSV exports. If leaked or accessed by rogue insiders, the system generates high-priority alerts.
* 🧅 **Onion Circuit Simulation:** The dashboard visualizes a dynamic, obfuscated network topology (Guard → Middle → Exit nodes) to disguise the underlying server architecture from attackers.
* 🧠 **Explainable AI (SHAP):** Transparently highlights exactly *why* a transaction was flagged by showing the feature importance for every prediction.
* 🔒 **Role-Based Access Control (RBAC):** Stateless JSON Web Token (JWT) authentication for secure segregation between Admins, Analysts, and Viewers.

---

## 🛠️ Technology Stack

### Frontend Architecture
* **HTML5 / CSS3:** Custom minimalist "glass-panel" dark-mode UI.
* **Vanilla JavaScript:** Framework-free asynchronous DOM manipulation.
* **WebSockets:** Low-latency live transaction streaming directly to the browser.

### Backend & API
* **Python 3:** Core logic and routing.
* **FastAPI & Uvicorn:** Lightning-fast, modern REST API and WebSocket handling.
* **SlowAPI:** Rate-limiting middleware to mitigate brute-force and DDoS attacks.
* **Passlib / Bcrypt:** Secure cryptographic password hashing.

### Machine Learning
* **LightGBM:** Tree-based learning algorithms.
* **Scikit-learn / PyTorch:** Autoencoder implementation for anomaly detection.
* **SHAP:** Explainable AI visualizations.

### Infrastructure & Deployment
* **Docker & Docker Compose:** Containerized microservices for isolated, seamless deployments.
* **Windows Batch Scripting:** One-click environment bootstrap and teardown.

---

## 🚀 Getting Started

### Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
* Windows OS (for the automated batch script launcher).

### Installation & Launch

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/deceptive-net.git
   cd deceptive-net
   ```

2. **Boot the Environment**
   Simply run the included boot sequence script. This script automatically checks Docker status, builds the isolated containers, and launches the application.
   ```cmd
   start_deceptive_net.bat
   ```

3. **Access the Application**
   The batch script will automatically open the dashboard in your default browser. 
   If it doesn't, navigate manually to:
   * **Dashboard:** `http://localhost:8080`
   * **Backend API Docs (Swagger):** `http://localhost:8000/docs`

### 🔑 Demo Credentials
* **Administrator:** `admin` / `admin123`
* **Analyst:** `analyst` / `analyst123`

---

## 🏗️ Architecture Flow

1. **Transaction Simulation:** The backend generates realistic synthetic financial data.
2. **Analysis:** The Ensemble Engine (LightGBM + Autoencoder) processes the data instantly.
3. **Distribution:** Scored transactions are pushed over WebSockets to the connected frontend clients.
4. **Deception:** If a user requests a secure export of the data, the backend injects a Honey Token and logs the exfiltration attempt into the system's Audit logs.

---

## ⚠️ Academic & Research Notice
**IMPORTANT:** This project is intended strictly for academic research and defensive cyber-security demonstration. 
* All transaction data processed by this system is mathematically synthesized.
* This project contains absolutely **no real financial data**, PII (Personally Identifiable Information), or deanonymization capabilities.

---

<div align="center">
  <small>&copy; copyright 2026 owner of the site is GeekyGeek.</small>
</div>
