"""P7 — Deployment boundary: cloud agent <-> local execution backend.

See alz_agent.boundary.transport for the transport protocol, the suppression
re-check guard, and the (de)serialization helpers.
"""

from alz_agent.boundary.transport import (
    BoundaryViolation,
    ToolCallRequest,
    deserialize_request,
    deserialize_tool_result,
    execute_tool_call,
    serialize_request,
    serialize_tool_result,
)

__all__ = [
    "BoundaryViolation",
    "ToolCallRequest",
    "execute_tool_call",
    "serialize_request",
    "deserialize_request",
    "serialize_tool_result",
    "deserialize_tool_result",
]
