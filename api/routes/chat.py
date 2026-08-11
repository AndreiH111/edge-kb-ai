import dataclasses
import json
from typing import Optional, Tuple
from uuid import uuid4

from fastapi import APIRouter, Body, Cookie, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse

from models.schemas import AgentStatusResponse, ChatRequest, ChatResponse
from services import agent_service


router = APIRouter(tags=["chat"])


def _resolve_question_and_session(
    request: Optional[ChatRequest],
    question: Optional[str],
    session_id: Optional[str],
    cookie_session_id: Optional[str],
) -> Tuple[Optional[str], str]:
    """Resolve the question and session id from body, query, or cookie.

    Precedence matches the original ``/chat`` handling: an explicit query
    parameter wins over the JSON body, which wins over the session cookie; a
    fresh uuid is minted when no session id is supplied.
    """
    user_question = question or (request.question if request else None)
    user_session_id = (
        session_id
        or (request.session_id if request else None)
        or cookie_session_id
        or str(uuid4())
    )
    return user_question, user_session_id


@router.post("/chat")
async def ask_question(
    response: Response,
    request: Optional[ChatRequest] = Body(default=None),
    question: Optional[str] = Query(default=None, min_length=1),
    session_id: Optional[str] = Query(default=None, min_length=1),
    cookie_session_id: Optional[str] = Cookie(default=None, alias="ai_session_id"),
) -> ChatResponse:
    user_question, user_session_id = _resolve_question_and_session(
        request, question, session_id, cookie_session_id
    )

    if not user_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide a question in the JSON body or as a query parameter.",
        )

    result = await agent_service.aanswer_question(user_question, user_session_id)
    response.set_cookie(
        key="ai_session_id",
        value=user_session_id,
        httponly=True,
        samesite="lax",
    )
    return ChatResponse(
        session_id=user_session_id,
        question=user_question,
        answer=result.answer,
        tools_used=result.tools_used or None,
    )


@router.post("/chat/stream")
async def ask_question_stream(
    request: Optional[ChatRequest] = Body(default=None),
    question: Optional[str] = Query(default=None, min_length=1),
    session_id: Optional[str] = Query(default=None, min_length=1),
    cookie_session_id: Optional[str] = Cookie(default=None, alias="ai_session_id"),
) -> StreamingResponse:
    """Stream the answer as Server-Sent Events.

    Emits ``event: token|tool|done|error`` frames, each with a JSON ``data``
    payload (JSON-escaping keeps embedded newlines from breaking the SSE frame).
    Non-breaking: the classic ``POST /chat`` is unchanged.
    """
    user_question, user_session_id = _resolve_question_and_session(
        request, question, session_id, cookie_session_id
    )

    if not user_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide a question in the JSON body or as a query parameter.",
        )

    async def event_stream():
        async for name, payload in agent_service.astream_answer(
            user_question, user_session_id
        ):
            yield f"event: {name}\ndata: {json.dumps(payload)}\n\n"

    response = StreamingResponse(event_stream(), media_type="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.set_cookie(
        key="ai_session_id",
        value=user_session_id,
        httponly=True,
        samesite="lax",
    )
    return response


@router.get("/agent-status")
def agent_status() -> AgentStatusResponse:
    """Report whether the agent and its Jira MCP tools loaded (diagnostics)."""
    return AgentStatusResponse(**dataclasses.asdict(agent_service.get_agent_status()))
