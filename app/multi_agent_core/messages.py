"""Structured messages exchanged through the supervisor."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


MessageStatus = Literal["pending", "running", "done", "failed", "retrying"]


class AgentMessage(BaseModel):
    task_id: str
    task_type: str
    from_agent: str = Field(alias="from")
    to_agent: str = Field(alias="to")
    content: dict[str, Any]
    status: MessageStatus = "pending"
    attempt: int = 0
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error_code: str | None = None

    model_config = {"populate_by_name": True}
