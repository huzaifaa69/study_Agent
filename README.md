# Study Agent — AI-Powered Study Assistant

A fully free RAG-based AI agent that lets you chat with your lecture PDFs,
generate MCQs, summaries, and flashcards using Groq's free LLM API.
**Working Link:**
https://studyagent001.streamlit.app/

---

## Project Structure

```
study_agent/
├── app.py                  ← Main Streamlit UI
├── backend/
│   ├── __init__.py
│   └── rag_engine.py       ← RAG logic (PDF → Chunks → Embeddings → Answer)
├── chroma_db/              ← Auto-created: stores your embeddings on disk
├── uploads/                ← Optional: drop PDFs here manually
├── requirements.txt
├── .env                    ← Your Groq API key goes here
└── README.md
```

---

## Setup — Step by Step

### Step 1: Get your FREE Groq API Key
1. Go to → https://console.groq.com
2. Sign up (free, no credit card)
3. Click "API Keys" → "Create API Key"
4. Copy the key

### Step 2: Add your API key
Open `.env` file and replace the placeholder:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

### Step 3: Create virtual environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 4: Install dependencies
```bash
pip install -r requirements.txt
```
> First install takes 3–5 minutes (downloads embedding model ~90MB)

### Step 5: Run the app
```bash
streamlit run app.py
```
> Opens at: http://localhost:8501

---

## How to Use

1. **Upload PDFs** — Drop your lecture notes/textbook PDFs in the sidebar
2. **Click "Process PDFs"** — Builds the knowledge base (30 sec first time)
3. **Chat Tab** — Ask any question, get answers with source references
4. **MCQ Tab** — Enter a topic → get exam-ready multiple choice questions
5. **Summary Tab** — Get structured notes with key concepts + exam tips
6. **Flashcards Tab** — Generate revision flashcards

---

## How It Works (RAG Pipeline)

```
Your PDF
   ↓
PDFPlumber extracts text
   ↓
RecursiveCharacterTextSplitter splits into 1000-char chunks
   ↓
all-MiniLM-L6-v2 converts chunks to embeddings (runs locally, FREE)
   ↓
ChromaDB stores embeddings on disk
   ↓
User asks question
   ↓
ChromaDB finds top 4 most relevant chunks (semantic search)
   ↓
Groq llama-3.3-70b-versatile generates answer using those chunks
   ↓
Answer + source references shown to user
```

---

## Everything is FREE

| Component | Tool | Cost |
|---|---|---|
| LLM (Brain) | Groq llama-3.3-70b-versatile | Free tier |
| Embeddings | all-MiniLM-L6-v2 | Free (local) |
| Vector DB | ChromaDB | Free (local) |
| UI | Streamlit | Free |
| PDF parsing | pdfplumber | Free |
| Hosting | Streamlit Cloud | Free |

---


