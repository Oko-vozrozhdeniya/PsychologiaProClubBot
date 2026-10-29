"""Граница между личным Telegram-диалогом и памятью Gemini.

Права доступа поступают только от проверенного внешнего провайдера.
При отсутствии подтверждённых прав обычный диалог продолжает работать
без долгосрочной памяти; никакого согласия автоматически не создаётся.
"""

from memory.prompt_context import build_memory_context
from memory.telegram_access import prepare_with_verified_access


def private_memory_for_gemini(
    *, user_id, message_id, text, timestamp, permission_provider, db_path=None
):
    """Return authorized memory prompt and source registration metadata.

    Fail closed: permission errors propagate to the caller, which may choose
    to continue an ordinary private chat without any memory context.
    """
    prepared = prepare_with_verified_access(
        user_id=user_id,
        message_id=message_id,
        text=text,
        timestamp=timestamp,
        permission_provider=permission_provider,
        db_path=db_path,
    )
    return {
        "prompt": build_memory_context(prepared["context"]),
        "source": prepared["source"],
    }
