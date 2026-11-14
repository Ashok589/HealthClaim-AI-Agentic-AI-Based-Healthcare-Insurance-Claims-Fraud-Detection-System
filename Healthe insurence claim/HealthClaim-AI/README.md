# HealthClaim AI 🏥

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Gemini](https://img.shields.io/badge/Google-Gemini_Pro-orange?logo=google)](https://ai.google.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](Dockerfile)

> **AI-powered health insurance claim processing system** — automated fraud detection, policy compliance checking, and intelligent adjudication using Google Gemini Pro and a multi-agent RAG pipeline.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **Multi-Agent Pipeline** | Orchestrated AI agents for document parsing, policy lookup, fraud detection, and final adjudication |
| 🔍 **Fraud Detection** | XGBoost + LLM ensemble with configurable score threshold |
| 📄 **OCR & Document Extraction** | PDF / image claim documents parsed via PyPDF2, pdfplumber, Tesseract |
| 📚 **RAG-Powered Policy Engine** | ChromaDB vector store + sentence-transformers for semantic policy retrieval |
| 🔐 **Role-Based Auth** | JWT authentication with admin / adjuster / auditor roles |
| 📊 **Streamlit Dashboard** | Interactive analytics and claim review UI |
| 🗄️ **SQLite / PostgreSQL** | SQLAlchemy ORM with Alembic migrations |
| 🐳 **Docker-Ready** | Single-command deployment |

---

## 🏗️ Architecture

```
                        ┌──────────────────────────────────┐
                        │        FastAPI REST API           │
                        └──────────────┬───────────────────┘
                                       │
              ┌────────────────────────▼──────────────────────────┐
              │              Agent Orchestrator                    │
              └───┬──────────────┬──────────────┬────────────┬────┘
                  │              │              │            │
           ┌──────▼──────┐ ┌────▼────┐ ┌──────▼─────┐ ┌───▼────────┐
           │  Document   │ │ Policy  │ │   Fraud    │ │  Decision  │
           │   Agent     │ │  Agent  │ │   Agent    │ │   Agent    │
           └──────┬──────┘ └────┬────┘ └──────┬─────┘ └───┬────────┘
                  │              │              │            │
           ┌──────▼──────────────▼──────────────▼────────────▼──────┐
           │          Shared Tools & RAG Layer                       │
           │  OCR · ChromaDB · Gemini Pro · XGBoost · SQLAlchemy     │
           └────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

- **Backend:** FastAPI, Uvicorn, SQLAlchemy 2.0, Alembic
- **AI / ML:** Google Gemini Pro, LangChain, XGBoost, scikit-learn, sentence-transformers
- **Vector DB:** ChromaDB
- **Document Processing:** PyPDF2, pdfplumber, Tesseract OCR, Pillow
- **Auth:** python-jose (JWT), passlib (bcrypt)
- **Frontend:** Streamlit, Plotly
- **Database:** SQLite (dev) / PostgreSQL (prod)
- **Infrastructure:** Docker

---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- Tesseract OCR (for image documents)
- Google AI API key

### 1. Clone & enter the project
```bash
git clone https://github.com/your-org/healthclaim-ai.git
cd healthclaim-ai
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env .env.local   # edit .env.local with your keys
```

Key variables to set:
| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Your Google Generative AI key |
| `SECRET_KEY` | Long random string for JWT signing |
| `DATABASE_URL` | SQLAlchemy connection string |

### 5. Initialise the database
```bash
python -c "from app.database.db import create_all_tables; create_all_tables()"
```

### 6. Start the API server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Launch the Streamlit dashboard (optional)
```bash
streamlit run frontend/dashboard.py
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/claims/` | Submit a new claim |
| `GET` | `/api/v1/claims/` | List claims (paginated) |
| `GET` | `/api/v1/claims/{id}` | Get claim details |
| `PATCH` | `/api/v1/claims/{id}` | Update claim status |
| `POST` | `/api/v1/claims/{id}/documents` | Upload supporting document |
| `POST` | `/api/v1/claims/{id}/process` | Trigger AI processing pipeline |
| `POST` | `/api/v1/policy/check` | Check policy coverage |
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/auth/login` | Obtain JWT token |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc |

---

## 🐳 Docker Deployment

### Build & run
```bash
docker build -t healthclaim-ai .
docker run -p 8000:8000 --env-file .env healthclaim-ai
```

### Docker Compose (recommended)
```yaml
version: "3.9"
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./chroma_db:/app/chroma_db
```

```bash
docker compose up -d
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v --tb=short
```

---

## 📂 Project Structure

```
HealthClaim-AI/
├── app/
│   ├── agents/          # AI agent implementations
│   ├── workflows/       # LangChain workflow definitions
│   ├── tools/           # Shared agent tools (OCR, DB, etc.)
│   ├── rag/             # RAG pipeline (ChromaDB + embeddings)
│   ├── models/          # ML model loaders / wrappers
│   ├── database/        # SQLAlchemy models & session
│   ├── schemas/         # Pydantic request / response schemas
│   ├── utils/           # Config, logger, helpers
│   └── main.py          # FastAPI application entry point
├── data/
│   ├── policies/        # Sample / seed policy JSON files
│   ├── claims/          # Uploaded claim document storage
│   └── medical_documents/ # ICD-10 / CPT code references
├── frontend/            # Streamlit dashboard
├── notebooks/           # Exploratory / training notebooks
├── tests/               # Pytest test suite
├── requirements.txt
├── Dockerfile
└── .env
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "feat: add my feature"`
4. Push to your fork: `git push origin feature/my-feature`
5. Open a Pull Request

Please ensure all tests pass and add docstrings to new public functions.

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
