
"""Безопасный адаптер личных сообщений для долгосрочной памяти.

Не подключён к bot.py. Не переносит старую историю автоматически.
"""

from datetime import datetime, timezone

from memory import lifecycle
from memory.coordinator import AccessPolicy, read_context


def register_private_message(
    user_id,
    message_id,
    content,
    timestamp,
    policy,
    *,
    db_path=None,
):
    """Сохраняет источник только при подтверждённых правах."""

    if (
        not isinstance(policy, AccessPolicy)
        or policy.user_id != user_id
    ):
        raise PermissionError(
            "Verified user policy required"
        )

    if not (
        policy.verified
        and policy.consent
        and policy.entitled
    ):
        raise PermissionError(
            "Consent and entitlement required"
        )

    if (
        not isinstance(timestamp, datetime)
        or timestamp.tzinfo is None
    ):
        raise ValueError(
            "Timezone-aware timestamp required"
        )

    now = timestamp.astimezone(timezone.utc)

    expiry = datetime.fromisoformat(
        policy.expires_at.replace("Z", "+00:00")
    )

    if expiry <= now:
        raise PermissionError("Permission expired")

    if type(message_id) is not int or message_id <= 0:
        raise ValueError("Invalid message ID")

    kwargs = (
        {"db_path": db_path}
        if db_path is not None
        else {}
    )

    lifecycle.initialize_lifecycle(**kwargs)

    source_id = (
        f"telegram:private:{user_id}:{message_id}"
    )

    expires_at = lifecycle.register_message(
        user_id,
        source_id,
        content,
        now,
        **kwargs,
    )

    return {
        "source_id": source_id,
        "expires_at": expires_at,
    }


def authorized_context(
    user_id,
    query,
    policy,
    permissions,
    timestamp,
    *,
    db_path=None,
):
    """Возвращает только разрешённую действующую память."""

    if (
        not isinstance(timestamp, datetime)
        or timestamp.tzinfo is None
    ):
        raise ValueError(
            "Timezone-aware timestamp required"
        )

    kwargs = (
        {"db_path": db_path}
        if db_path is not None
        else {}
    )

    return read_context(
        user_id,
        query,
        policy,
        permissions,
        timestamp.isoformat(),
        enforce_lifecycle=True,
        **kwargs,
    )
