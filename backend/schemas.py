from pydantic import BaseModel
from typing import List, Optional
import datetime


class UserCreate(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class DocumentOut(BaseModel):
    id: int
    filename: str
    num_chunks: int
    uploaded_at: datetime.datetime

    class Config:
        from_attributes = True


class ChatSessionOut(BaseModel):
    id: int
    title: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class SourceSnippet(BaseModel):
    source: str
    page: Optional[int] = None
    snippet: str


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    sources: Optional[List[SourceSnippet]] = None
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    session_id: Optional[int] = None
    query: str


class ChatResponse(BaseModel):
    session_id: int
    answer: str
    sources: List[SourceSnippet]
