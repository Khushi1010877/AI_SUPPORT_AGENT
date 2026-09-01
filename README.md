# AI Customer Support Agent

AI-powered support agent that answers questions from your own PDFs using RAG.

## Features

- Login and JWT authentication
- PDF upload and document search
- Conversational chat with source citations
- Persistent chat history per user
- Groq-powered answers
- Local ChromaDB embeddings, so OpenAI is not required

## Setup

```powershell
git clone https://github.com/Khushi1010877/AI_SUPPORT_AGENT.git
cd AI_SUPPORT_AGENT
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

```env
SECRET_KEY=replace-with-a-long-random-secret
LLM_PROVIDER=groq
GROQ_API_KEY=your-groq-api-key
GROQ_CHAT_MODEL=openai/gpt-oss-20b
```


## Run

Open two terminals from the project root.

Terminal 1, backend:
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

Terminal 2, frontend:
```bash
cd frontend
python -m streamlit run app.py
```

Open the Streamlit URL it prints (usually http://localhost:8501), register an
account, upload a PDF, and start chatting.

The backend API documentation is available at http://localhost:8000/docs.

## Security

- Never commit `.env` or expose API keys.
- Use a strong, unique `SECRET_KEY` outside local development.
- Uploaded documents and local databases are excluded from Git by `.gitignore`.

