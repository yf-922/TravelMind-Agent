"""Minimal protocol objects mirroring MCP tool discovery and A2A task exchange.

These are local, dependency-free contracts. They are intentionally not called
an MCP/A2A server: no network transport or remote discovery is implemented.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentCard(BaseModel):
    name: str
    description: str
    skills: list[str] = Field(default_factory=list)
    accepted_task_types: list[str] = Field(default_factory=list)


class MCPToolManifest(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    allowed_agents: list[str] = Field(default_factory=list)


class A2ATaskEnvelope(BaseModel):
    task_id: str
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    sender: str
    recipient: str
    task_type: str
    payload: dict[str, Any]
    status: Literal["submitted", "working", "completed", "failed"] = "submitted"
    attempt: int = Field(default=0, ge=0)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def agent_card(name: str, description: str, *, skills: list[str] | None = None,
               accepted_task_types: list[str] | None = None) -> AgentCard:
    return AgentCard(
        name=name,
        description=description,
        skills=list(skills or []),
        accepted_task_types=list(accepted_task_types or []),
    )


def a2a_from_message(message: Any) -> A2ATaskEnvelope:
    """Adapt the existing local AgentMessage into an A2A-shaped envelope."""
    return A2ATaskEnvelope(
        task_id=message.task_id,
        trace_id=message.trace_id,
        sender=message.from_agent,
        recipient=message.to_agent,
        task_type=message.task_type,
        payload=dict(message.content),
        status="completed" if message.status == "done" else (
            "failed" if message.status == "failed" else "working"
        ),
        attempt=message.attempt,
        created_at=message.created_at,
    )
