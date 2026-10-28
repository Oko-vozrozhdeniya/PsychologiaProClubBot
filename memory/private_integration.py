
"""Безопасная точка подключения личной памяти Telegram-бота.

Не предоставляет согласие или права автоматически.
Не изменяет обработчики действующего bot.py.
"""

from datetime import datetime, timezone

from memory import lifecycle, storage
from memory.coordinator import AccessPolicy, SourcePermission
from memory.dialog_bridge import (
    authorized_context,
    register_private_message,
)


def prepare_private_memory(
    *,
    user_id,
    message_id,
    text,
    timestamp,
    policy,
    source_permissions,
    db_path=None,
):
    """Регистрирует разрешённое сообщение и читает доступную память."""

    if type(user_id) is not int or user_id <= 0:
        raise ValueError("Invalid Telegram user")

    if type(message_id) is not int or message_id <= 0:
        raise ValueError("Invalid Telegram message")

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Message text required")

    if not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
        raise ValueError("Timezone-aware timestamp required")

    if not isinstance(policy, AccessPolicy) or policy.user_id != user_id:
        raise PermissionError("Verified matching policy required")

    if not (
        policy.verified
        and policy.consent
        and policy.entitled
    ):
        raise PermissionError(
            "Explicit consent and entitlement required"
        )

    if not isinstance(source_permissions, (list, tuple)) or not all(
        isinstance(item, SourcePermission)
        for item in source_permissions
    ):
        raise PermissionError("Verified source permissions required")

    # Проверяем срок действия разрешения до создания базы данных.
    now = timestamp.astimezone(timezone.utc)

    try:
        expiry = datetime.fromisoformat(
            policy.expires_at.replace("Z", "+00:00")
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise PermissionError("Invalid permission expiry") from exc

    if expiry.tzinfo is None or expiry <= now:
        raise PermissionError("Permission expired")

    # Инициализируем обе части базы памяти.
    database = (
        db_path
        if db_path is not None
        else storage.DATABASE_PATH
    )

    storage.initialize_database(db_path=database)
    lifecycle.initialize_lifecycle(db_path=database)

    # Проверяем доступ к памяти до сохранения сообщения.
    authorized_context(
        user_id,
        text,
        policy,
        source_permissions,
        timestamp,
        db_path=db_path,
    )

    registered = register_private_message(
        user_id,
        message_id,
        text,
        timestamp,
        policy,
        db_path=db_path,
    )

    context = authorized_context(
        user_id,
        text,
        policy,
        source_permissions,
        timestamp,
        db_path=db_path,
    )

    return {
        "source": registered,
        "context": context,
    }
