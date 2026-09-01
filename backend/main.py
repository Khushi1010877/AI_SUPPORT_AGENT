import json
import os
import shutil

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from config import settings
from database import init_db, get_db, User, Document, ChatSession, Message
from schemas import (
    UserCreate, Token, DocumentOut, ChatSessionOut, MessageOut,
    ChatRequest, ChatResponse, SourceSnippet,
)
from auth import (
    hash_password, authenticate_user, create_access_token, get_current_user,
)
import rag

app = FastAPI(title=settings.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


# ---------------------------------------------------------------- Auth ----

@app.post("/register", response_model=Token)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == user_in.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")
    user = User(username=user_in.username, hashed_password=hash_password(user_in.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token({"sub": user.username})
    return Token(access_token=token)


@app.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token({"sub": user.username})
    return Token(access_token=token)


# ----------------------------------------------------------- Documents ----

@app.post("/upload", response_model=DocumentOut)
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    user_dir = os.path.join(settings.UPLOAD_DIR, str(current_user.id))
    os.makedirs(user_dir, exist_ok=True)
    dest_path = os.path.join(user_dir, file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        num_chunks = rag.ingest_pdf(dest_path, current_user.id, file.filename)
    except rag.DocumentIngestionError as e:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    doc = Document(user_id=current_user.id, filename=file.filename, num_chunks=num_chunks)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    return db.query(Document).filter(Document.user_id == current_user.id).all()


@app.delete("/documents/{doc_id}")
def delete_document(
    doc_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    doc = db.query(Document).filter(
        Document.id == doc_id, Document.user_id == current_user.id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    rag.delete_document(current_user.id, doc.filename)
    file_path = os.path.join(settings.UPLOAD_DIR, str(current_user.id), doc.filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    db.delete(doc)
    db.commit()
    return {"detail": "deleted"}


# ------------------------------------------------------------- Sessions ----

@app.post("/sessions", response_model=ChatSessionOut)
def create_session(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    session = ChatSession(user_id=current_user.id, title="New Chat")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@app.get("/sessions", response_model=list[ChatSessionOut])
def list_sessions(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    return (
        db.query(ChatSession)
        .filter(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )


@app.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
def get_session_messages(
    session_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    session = _get_owned_session(db, session_id, current_user)
    out = []
    for m in session.messages:
        sources = json.loads(m.sources) if m.sources else None
        out.append(MessageOut(
            id=m.id, role=m.role, content=m.content, sources=sources, created_at=m.created_at
        ))
    return out


@app.delete("/sessions/{session_id}")
def delete_session(
    session_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    session = _get_owned_session(db, session_id, current_user)
    db.delete(session)
    db.commit()
    return {"detail": "deleted"}


def _get_owned_session(db: Session, session_id: int, user: User) -> ChatSession:
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id, ChatSession.user_id == user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return session


# ----------------------------------------------------------------- Chat ----

@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    # Get or create session
    if req.session_id:
        session = _get_owned_session(db, req.session_id, current_user)
    else:
        session = ChatSession(user_id=current_user.id, title=req.query[:50])
        db.add(session)
        db.commit()
        db.refresh(session)

    # Build history (excluding the current query)
    history = [(m.role, m.content) for m in session.messages]

    try:
        answer, source_docs = rag.answer_query(current_user.id, req.query, history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")

    sources = [
        SourceSnippet(
            source=d.metadata.get("source", "unknown"),
            page=d.metadata.get("page"),
            snippet=d.page_content[:300],
        )
        for d in source_docs
    ]

    # Persist user + assistant messages
    db.add(Message(session_id=session.id, role="user", content=req.query))
    db.add(Message(
        session_id=session.id,
        role="assistant",
        content=answer,
        sources=json.dumps([s.model_dump() for s in sources]),
    ))
    if session.title == "New Chat":
        session.title = req.query[:50]
    db.commit()

    return ChatResponse(session_id=session.id, answer=answer, sources=sources)


@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": settings.LLM_PROVIDER}
