"""Shared classifier for LLM provider errors.

Returns a stable ``error_type`` string used by both the agent-side server
LLM loop (``terralingua/server/api.py``) and the anthropologist server. Keeping
the string set consistent lets the frontend's existing buffering in
``addStepError()`` coalesce errors across both sources when the same key
is hitting the same limit.
"""


def llm_error_type(exc: Exception) -> str | None:
    """Returns a stable type string for API errors worth surfacing, else None."""
    if not hasattr(exc, "status_code"):
        return None
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        # Providers nest the type under "error" or put it at the top level, and
        # "error" may be a plain string.
        error = body.get("error")
        sdk_type = error.get("type") if isinstance(error, dict) else body.get("type")
        if isinstance(sdk_type, str) and sdk_type:
            return sdk_type
    status = getattr(exc, "status_code", None)
    return f"http_{status}" if status else "api_error"
