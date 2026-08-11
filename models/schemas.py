from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = Field(default=None, min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    tools_used: Optional[List[str]] = None


class AgentStatusResponse(BaseModel):
    """Inspectable state of the agent built during the app lifespan.

    Mirrors ``agent_service.AgentStatus`` so ``GET /ai/agent-status`` can surface
    whether the agent and its Jira MCP tools actually loaded (instead of silently
    degrading to a retrieval-only or legacy chain).
    """

    agent_enabled: bool
    agent_ready: bool
    jira_enabled: bool
    jira_loaded: bool
    tool_count: int
    tools: List[str] = Field(default_factory=list)
    jira_error: Optional[str] = None


class FileUploadResponse(BaseModel):
    filename: str
    path: str
    indexed_records: int


class ReindexResponse(BaseModel):
    indexed_records: int
    files: List[str]


class FileInfo(BaseModel):
    filename: str
    uploaded_by: Optional[str] = None
    uploaded_at: Optional[str] = None


class FileListResponse(BaseModel):
    destination: str
    location: str
    files: List[FileInfo]


class DeleteFilesResponse(BaseModel):
    deleted: List[str]
    not_found: List[str]
    indexed_records: int
    message: str

class DeleteFilesRequest(BaseModel):
    filenames: List[str] = Field(..., min_length=1)


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    type: str
    filename: Optional[str] = None
    error: Optional[str] = None
    indexed_records: Optional[int] = None
    created_at: str
    completed_at: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: List[JobStatusResponse]


class UploadAcceptedResponse(BaseModel):
    job_id: str
    filename: str
    path: str
    status: str = "queued"


class ReindexAcceptedResponse(BaseModel):
    job_id: str
    status: str = "queued"