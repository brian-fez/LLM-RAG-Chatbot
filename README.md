# 🤖 Document RAG Chat

Document RAG Chat is a lightweight RAG-based chatbot built with **Streamlit** that allows you to upload and chat with any PDF document. It uses **HuggingFace embeddings**, a **FAISS vector store**, and a **local LLM** (like Ollama's deepseek-r1:8b) to provide accurate and contextual answers.

---

## ✨ Features

- 📂 Upload a PDF and ask questions about its content
- 🔍 Uses semantic search over embedded chunks
- 💬 Chat interface with session history and saving
- 🧠 Hidden reasoning (`<think>` sections) logged but not shown in UI
- 🧹 "New Chat" clears chat, state, and uploaded documents

---

## 📁 Project Structure

```
document-rag-chat/
│
├── app.py              # Main Streamlit app
├── chatbot.py          # Chatbot logic (LLM + vector store)
├── vectors.py          # PDF loading, chunking, embedding
├── utilities.py        # Helpers: session, logging, cleanup
│
├── chat_sessions/      # Saved chat history (.json)
├── chat_logs/          # Full LLM logs with reasoning
├── temp/               # Temporary PDF storage
│
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/brian-fez/LLM-RAG-Chatbot.git
   cd LLM-RAG-Chatbot
   ```

2. **(Optional) Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the App

To start the chatbot interface:

```bash
streamlit run app.py
```

Then open your browser to the URL shown (usually `http://localhost:8501`).

---

## 🧠 How It Works

1. **PDF Upload**: You upload a document.
2. **Embedding**: The text is split into chunks and embedded with a HuggingFace model (e.g., `BAAI/bge-small-en-v1.5`).
3. **Vector Store**: The chunks are saved in a FAISS index.
4. **Retrieval + Generation**:
   - Relevant chunks are retrieved based on your question.
   - An LLM (local or remote) generates a response using the chunks as context.
5. **Display**:
   - The full output (including `<think>` hidden reasoning) is logged to disk.
   - Only the clean, visible part is shown in the UI.

---

## 💾 Sessions & Logs

- 💬 Chat messages are saved in `/chat_sessions/` as `.json`.
- 🧠 Full LLM outputs are saved in `/chat_logs/` with timestamps.
- 📂 Uploaded PDFs are temporarily stored in `/temp/` and removed when starting a new chat.

---

## ✅ Requirements

- Python 3.9+
- Streamlit
- FAISS
- HuggingFace Transformers & Sentence Transformers
- Optional: Ollama (for local LLM like `deepseek`)

See `requirements.txt` for full dependencies.

---

## 🧼 Future Improvements

- 🗂️ Multi-document support
- 🛡️ Authentication
- 🌍 Deploy to Streamlit Cloud or Hugging Face Spaces
- 🐳 Docker container for production
- 💬 Support chat history export

---