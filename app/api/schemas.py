"""API request/response schemas (Pydantic, reused by FastAPI docs)."""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field

from app.memory.models import MessageRecord, SessionInfo

#: Session ids are short hex strings (readable in URLs).
_DEFAULT_SESSION_LENGTH = 12


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)
    session_id: str | None = Field(
        default=None,
        description="Session à continuer; omit pour en créer une nouvelle.",
    )

    def resolved_session(self) -> str:
        """Existing session id, or a fresh one."""
        return self.session_id or uuid4().hex[:_DEFAULT_SESSION_LENGTH]


class ChatResponse(BaseModel):
    session_id: str
    response: str
    tools_used: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    provider: str
    llm_available: bool
    ollama_available: bool
    model: str
    workspace: str
    database_path: str
    tools: int
    mcp_servers: int
    sessions: int


class SessionSummary(BaseModel):
    id: str
    created_at: str
    updated_at: str
    message_count: int

    @classmethod
    def from_session(cls, info: SessionInfo) -> SessionSummary:
        return cls(
            id=info.id,
            created_at=info.created_at.isoformat(),
            updated_at=info.updated_at.isoformat(),
            message_count=info.message_count,
        )


class MessageOut(BaseModel):
    id: int | None
    role: str
    content: str
    created_at: str

    @classmethod
    def from_record(cls, record: MessageRecord) -> MessageOut:
        return cls(
            id=record.id,
            role=record.role.value,
            content=record.content,
            created_at=record.created_at.isoformat(),
        )


class SessionDetail(BaseModel):
    session: SessionSummary
    messages: list[MessageOut]


class DeleteResponse(BaseModel):
    deleted: bool
    session_id: str


class MCPServerOut(BaseModel):
    name: str
    transport: str
    state: str
    permission: str
    error: str | None = None
    tools: list[str] = Field(default_factory=list)


class UploadedFileOut(BaseModel):
    name: str
    path: str
    size: int
    extracted_path: str | None = None
