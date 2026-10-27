
"""Rules for safely updating long-term memory records.

This module prepares update decisions. It does not modify the database
automatically and does not call Gemini.
"""

from dataclasses import dataclass
from typing import Optional

VALID_ACTIONS = frozenset({
    "keep",
    "update",
    "conflict",
    "review",
})


@dataclass(frozen=True)
class UpdateDecision:
    action: str
    reason: str
    old_content: str
    new_content: str
    source: str


def _required_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def make_update_decision(
    old_content,
    new_content,
    source,
    *,
    explicitly_corrected=False,
    conflict_detected=False,
):
    """Prepare a decision without writing to persistent storage.

    An explicit correction may be applied only when no conflict is flagged.
    Other meaningful changes require review.
    """
    old_content = _required_text(old_content, "old_content")
    new_content = _required_text(new_content, "new_content")
    source = _required_text(source, "source")

    if not isinstance(explicitly_corrected, bool):
        raise ValueError("explicitly_corrected must be boolean")

    if not isinstance(conflict_detected, bool):
        raise ValueError("conflict_detected must be boolean")

    if old_content == new_content:
        action = "keep"
        reason = "Content is unchanged"
    elif conflict_detected:
        action = "conflict"
        reason = "Conflicting information requires review"
    elif explicitly_corrected:
        action = "update"
        reason = "Explicit correction supplied"
    else:
        action = "review"
        reason = "Change requires confirmation"

    return UpdateDecision(
        action=action,
        reason=reason,
        old_content=old_content,
        new_content=new_content,
        source=source,
    )


def decision_to_record(decision):
    """Convert an update decision to a plain dictionary."""
    if not isinstance(decision, UpdateDecision):
        raise ValueError("decision must be an UpdateDecision")

    return {
        "action": decision.action,
        "reason": decision.reason,
        "old_content": decision.old_content,
        "new_content": decision.new_content,
        "source": decision.source,
    }
