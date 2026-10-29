"""Isolated Telegram live-test entrypoint for the memory module.

Deploy to a separate, stopped Bothost service from the feature branch.
This service can use the existing Telegram bot identity only with an explicit
opt-in; the original service MUST be stopped before starting test polling.
Its application registers private handlers only and uses isolated storage.
Never commit Telegram or AI tokens.
"""
import asyncio
import os
from pathlib import Path

TEST_DATA_DIR = Path("/app/data/psychologia_live_test")
EXPECTED_PRODUCTION_USERNAME = "psychologiaproclub_bot"


def configure_test_runtime():
    """Fail closed before importing bot.py or creating any data."""
    if os.environ.get("BOT_TEST_MODE") != "1":
        raise RuntimeError("BOT_TEST_MODE=1 is required for test_bot.py")
    raw_username = os.environ.get("TEST_BOT_USERNAME", "").strip()
    expected_username = raw_username.lstrip("@").lower()
    if not expected_username or not expected_username.endswith("bot"):
        raise RuntimeError("Set the confirmed TEST_BOT_USERNAME")
    if expected_username == EXPECTED_PRODUCTION_USERNAME:
        if os.environ.get("TEST_BOT_USE_EXISTING_IDENTITY") != "1":
            raise RuntimeError(
                "Using the existing Telegram bot requires "
                "TEST_BOT_USE_EXISTING_IDENTITY=1 and stopping the other "
                "BotHost service before starting this one"
            )
    elif os.environ.get("TEST_BOT_USE_EXISTING_IDENTITY") == "1":
        raise RuntimeError(
            "TEST_BOT_USE_EXISTING_IDENTITY=1 is only for the existing bot"
        )
    raw_ids = os.environ.get("TEST_BOT_ALLOWED_USER_IDS", "").strip()
    if (not raw_ids or any(not item.strip().isdigit()
            or int(item.strip()) <= 0 for item in raw_ids.split(","))):
        raise RuntimeError("Set TEST_BOT_ALLOWED_USER_IDS to numeric user IDs")
    allowed = {int(item.strip()) for item in raw_ids.split(",")}
    if not os.environ.get("BOT_TOKEN", "").strip():
        raise RuntimeError("BOT_TOKEN must be configured on the stopped test service")
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        raise RuntimeError("GEMINI_API_KEY is required for live AI testing")
    # No service may override the isolated data paths with a production path.
    expected_data = str(TEST_DATA_DIR)
    expected_db = str(TEST_DATA_DIR / "memory.db")
    for name, expected in (("BOT_DATA_DIR", expected_data),
                           ("MEMORY_DB_PATH", expected_db)):
        configured = os.environ.get(name)
        if configured is not None and configured != expected:
            raise RuntimeError(f"{name} must be {expected} in test mode")
        os.environ[name] = expected
    return expected_username, allowed


async def verify_test_bot_identity(application, expected_username):
    """Check BotFather identity before Telegram polling starts."""
    me = await application.bot.get_me()
    actual = (me.username or "").lower()
    if actual != expected_username:
        raise RuntimeError(
            "Wrong Telegram bot identity. Test polling was not started."
        )
    if (actual == EXPECTED_PRODUCTION_USERNAME and
            os.environ.get("TEST_BOT_USE_EXISTING_IDENTITY") != "1"):
        raise RuntimeError("Existing bot identity not explicitly authorized")


async def start_test(update, context):
    await update.effective_message.reply_text(
        "Это тестовый режим памяти «Психология PRO».\n"
        "Сначала /memory_on — согласие на тестовую память.\n"
        "Затем /remember <текст>, /memories, /correct <номер> <новый текст>, "
        "/change <номер> <новый текст>, /history <номер>.\n"
        "Проверка извлечения: /check_memory <слова из факта>.\n"
        "Остановить использование памяти: /memory_off.\n"
        "Пожалуйста, используйте вымышленные данные при проверке."
    )


async def check_memory(update, context):
    import bot
    query = " ".join(context.args).strip()
    if not query:
        await update.effective_message.reply_text(
            "Формат: /check_memory слова_из_сохранённого_факта"
        )
        return
    result = await asyncio.to_thread(
        bot.read_explicit_memory_prompt, update.effective_user.id, query
    )
    if not result:
        result = "Подходящих разрешённых воспоминаний не найдено."
    await bot.send_long_message(update.effective_message, result)


def create_test_application(bot, allowed_ids, expected_username):
    """Private-only routes; group/archive/topic handlers do not exist here."""
    from telegram.ext import Application, CommandHandler, MessageHandler, filters

    if not allowed_ids.issubset({bot.OWNER_ID, *bot.TEST_USER_IDS}):
        raise RuntimeError(
            "Test user IDs must already be authorized in the feature bot.py"
        )
    private_tester = (
        filters.ChatType.PRIVATE & filters.User(user_id=sorted(allowed_ids))
    )
    app = (Application.builder().token(bot.TOKEN).post_init(
        lambda application: verify_test_bot_identity(
            application, expected_username
        )
    ).build())
    commands = {
        "start": start_test,
        "memory_on": bot.memory_on_command,
        "memory_off": bot.memory_off_command,
        "remember": bot.remember_command,
        "memories": bot.memories_command,
        "forget": bot.forget_command,
        "correct": bot.correct_command,
        "change": bot.change_command,
        "history": bot.history_command,
        "check_memory": check_memory,
    }
    for name, handler in commands.items():
        app.add_handler(CommandHandler(
            name, handler, filters=private_tester
        ))
    app.add_handler(MessageHandler(
        private_tester & filters.TEXT & ~filters.COMMAND,
        bot.private_message,
    ))
    return app


def main():
    username, allowed_ids = configure_test_runtime()
    import bot  # must happen AFTER storage paths were isolated
    import memory.storage as storage
    if (bot.DATA_DIR != TEST_DATA_DIR or
            storage.DATABASE_PATH != TEST_DATA_DIR / "memory.db"):
        raise RuntimeError("Test and production storage paths are not isolated")
    application = create_test_application(bot, allowed_ids, username)
    print(f"Starting isolated memory test bot @{username}", flush=True)
    application.run_polling()


if __name__ == "__main__":
    main()
