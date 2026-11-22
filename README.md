# 🌿 PlantCare AI — Autonomous Foliar Pathology Intelligence

PlantCare AI is an enterprise-grade agricultural vision system engineered to detect, classify, and diagnose 38 botanical diseases across 14 commercial crop varieties with Bayesian out-of-distribution guardrails, actionable curative treatments, and long-term prevention protocols.

Styled with the **xAI Frontier Design System** (monochrome `#0a0a0a` canvas, `#ff7a17` amber accent, pill navigation, and ultra-crisp typography).

---

## ⚡ Quick Start

### 1. Prerequisites & Environment Setup
Clone the repository and create a Python virtual environment:
```bash
git clone https://github.com/<YOUR_USERNAME>/PlantCareAI.git
cd PlantCareAI

python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Launch Development Server
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌟 Key Capabilities

* **🔬 38-Class Botanical Taxonomy**:
  Indexes 38 foliar conditions across Apple, Blueberry, Cherry, Corn, Grape, Citrus, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.
* **🏥 Curative & Preventive Agronomy**:
  Every diagnosis outputs an immediate curative intervention protocol (bio-fungicides, pruning, spray schedules) paired with cultural field prevention strategies (drip irrigation, plant spacing, crop rotation).
* **🧠 Real Leaf Specimen Library**:
  All 38 conditions feature verified specimen photography loaded directly from in-memory cache without external asset dependencies.
* **⚡ Sub-Millisecond Local-First Architecture**:
  Queries execute against in-process SQLite in `<2 ms` with automatic background synchronization to Supabase PostgreSQL.
* **🔒 Enterprise Security & Guardrails**:
  PBKDF2-HMAC-SHA256 password hashing, role-based access control (Admin Panel restricted to verified administrators), magic-byte file upload validation, and ReportLab XML injection defenses.

---

## 🧪 Automated Test Suite

Run the full automated test suite (52 tests across unit, integration, and security checks):
```bash
python -m unittest test_prediction_guard.py test_security.py test_app_integration.py test_end_to_end_verification.py
```

---

## 🚀 Cloud Deployment Options

### Option 1: Streamlit Community Cloud (Recommended & Free)
Streamlit Cloud is the official, native platform for Streamlit applications with full support for persistent WebSockets and PyTorch models:
1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"**, select your repository, branch (`main`), and set Main file path to `app.py`.
4. Add your secrets in **App settings > Secrets** if connecting to Supabase:
   ```toml
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_KEY = "your-anon-or-service-key"
   ```
5. Click **Deploy!**

### Option 2: Hugging Face Spaces / Render
* **Hugging Face Spaces**: Create a new Space, choose **Streamlit** SDK, and link your GitHub repository.
* **Render**: Deploy as a Web Service running `streamlit run app.py --server.port 10000 --server.address 0.0.0.0`.

---

## 📄 License
MIT License. Built for agricultural health research and grower productivity.
