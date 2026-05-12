# 🏥 Agentic AI Medical Assistant (RAG with DeepSeek R1)

A RAG-powered medical document chatbot built with LangChain, FAISS, DeepSeek R1 via Groq, and Streamlit.
Ask questions about uploaded patient records, lab reports, and clinical documents — with **zero hallucinations**.

---

## 🚀 Features

- 📄 **Multi-PDF Upload** — medical records, lab reports, prescriptions, clinical notes
- 🧠 **RAG Pipeline** — FAISS vector retrieval + LLM generation
- 🔒 **Hallucination-Reduced** — strict context-grounded answering
- 📚 **Source Citations** — every answer cites the exact document & page
- ⚡ **DeepSeek R1 via Groq** — blazing-fast inference (free tier available)
- 🎛️ **Configurable** — model selection, temperature, retrieval k

---

## 🛠️ Tech Stack

| Component     | Technology                              |
|---------------|-----------------------------------------|
| LLM           | DeepSeek R1 Distill 70B (via ChatGroq)  |
| Embeddings    | sentence-transformers/all-MiniLM-L6-v2  |
| Vector Store  | FAISS (CPU)                             |
| RAG Framework | LangChain                               |
| UI            | Streamlit                               |
| PDF Loader    | PyPDF (LangChain Community)             |

---

## ⚙️ Setup & Run

### 1. Clone / Download the project

```bash
cd medical_assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Groq API Key

```bash
cp .env.example .env
# Edit .env and paste your Groq API key
```

Or just paste it in the sidebar when the app opens.
> **Get a free Groq API key:** https://console.groq.com

### 5. Run the app

```bash
streamlit run app.py
```

---

## 💬 Example Questions

- *"What medications is the patient currently on?"*
- *"What were the CBC blood test results?"*
- *"Summarise the patient's medical history."*
- *"Are there any drug allergies mentioned?"*
- *"What diagnosis was given on the last visit?"*

---

## ⚠️ Disclaimer

This tool is for **informational and educational purposes only**.
Always consult a licensed physician for medical decisions.

---

## 📁 Project Structure

```
medical_assistant/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md           # This file
```
