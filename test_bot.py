"""Isolated Telegram live-test entrypoint for the memory module.

Deploy to a separate, stopped Bothost service from the feature branch.
It uses the already configured Telegram bot identity, but the original
service MUST be stopped before starting test polling.
Non-secret test settings are pinned here to avoid lost Bothost form inputs.
Its application registers private handlers only and uses isolated storage.
Never commit Telegram or AI tokens.
"""
import asyncio
import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

TEST_DATA_DIR = Path("/app/data/psychologia_live_test")
EXPECTED_PRODUCTION_USERNAME = "psychologiaproclub_bot"


def configure_test_runtime():
    """Configure strict test-only isolation using safe, nonsecret defaults.

    Only the existing bot identity and the repository's known owner are
    permitted. Missing optional Bothost UI variables cannot silently disable
    the test. Explicitly conflicting values fail closed.
    """
    if os.environ.get("BOT_TEST_MODE", "1") != "1":
        raise RuntimeError("Test mode cannot be disabled for test_bot.py")

    expected_username = os.environ.get(
        "TEST_BOT_USERNAME", EXPECTED_PRODUCTION_USERNAME
    ).strip().lstrip("@").lower()
    if expected_username != EXPECTED_PRODUCTION_USERNAME:
        raise RuntimeError("Unexpected test Telegram bot username")
    if os.environ.get("TEST_BOT_USE_EXISTING_IDENTITY", "1") != "1":
        raise RuntimeError("Existing identity must be explicitly enabled")

    raw_ids = os.environ.get("TEST_BOT_ALLOWED_USER_IDS", "885005330")
    if not raw_ids or any(
        not part.strip().isdigit() or int(part.strip()) <= 0
        for part in raw_ids.split(",")
    ):
        raise RuntimeError("Test user list must contain valid numeric IDs")
    allowed = {int(part.strip()) for part in raw_ids.split(",")}
    if allowed != {885005330}:
        raise RuntimeError("Only the preapproved owner can test this bot")

    if not os.environ.get("BOT_TOKEN", "").strip():
        raise RuntimeError("BOT_TOKEN must be configured in Bothost")
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        raise RuntimeError("GEMINI_API_KEY must remain configured in Bothost")

    # Never allow an optional environment override to point into live data.
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


async def start_test(update, context):
    await update.effective_message.reply_text(
        "Это тестовый режим памяти «Психология PRO».\n"
        "Сначала /memory_on — согласие на тестовую память.\n"
        "Затем /remember <текст>, /memories, /correct <номер> <новый текст>, "
        "/change <номер> <новый текст>, /history <номер>.\n"
        "Проверка извлечения: /check_memory <слова из факта>.\n"
        "Диагностика страны сервера: /network_check.\n"
        "Проверка доступности Gemini API: /gemini_check.\n"
        "Короткая пробная генерация Gemini: /gemini_ping.\n"
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


def detect_server_country():
    """Check external egress country only; never expose the public IP.

    This third-party lookup is for the owner's explicit diagnostic request.
    The returned country is informative: Google may classify an IP differently.
    """
    request = urllib.request.Request(
        "https://ipinfo.io/country",
        headers={"User-Agent": "PsychologiaPro-MemoryTest/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        # A country endpoint must return just a 2-letter country code.
        raw = response.read(16).decode("ascii").strip()
    if len(raw) != 2 or not raw.isalpha() or not raw.isupper():
        raise ValueError("Unexpected country lookup response")
    return raw


async def network_check(update, context):
    """Owner-only country check, isolated from the Gemini request path."""
    try:
        code = await asyncio.to_thread(detect_server_country)
    except (OSError, ValueError, UnicodeError):
        await update.effective_message.reply_text(
            "Не удалось определить страну выходного IP через внешний сервис. "
            "Это не влияет на сохранённую память."
        )
        return
    await update.effective_message.reply_text(
        f"Страна выхода сервера по ipinfo.io: {code}.\n"
        "Это оценка стороннего сервиса, а не решение Google "
        "о доступности Gemini API."
    )


def probe_gemini_models():
    """Check Gemini model-list endpoint with existing key; never expose secrets.

    No generateContent request is made and no private conversation is sent.
    Only HTTP code and sanitized Google status are returned.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return "Ошибка конфигурации: GEMINI_API_KEY отсутствует"
    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1",
        headers={"x-goog-api-key": api_key},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            data = json.loads(response.read(8192).decode("utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            return "HTTP 200: неожиданный формат ответа Google"
        return "HTTP 200 OK: Gemini API принимает ключ и возвращает список моделей"
    except urllib.error.HTTPError as error:
        google_status = "UNKNOWN"
        google_message = ""
        try:
            info = json.loads(error.read(4096).decode("utf-8"))
            details = info.get("error", {}) if isinstance(info, dict) else {}
            if isinstance(details, dict):
                value = details.get("status")
                if (isinstance(value, str) and len(value) <= 50
                        and value.replace("_", "").isupper()
                        and value.replace("_", "").isalpha()):
                    google_status = value
                google_message = str(details.get("message", ""))[:250]
        except (ValueError, UnicodeError, OSError):
            pass
        if ("location" in google_message.lower() and
                "not supported" in google_message.lower()):
            category = "Google отклоняет расположение запроса"
        elif error.code in (401, 403):
            category = "Google отклоняет доступ по ключу или проекту"
        elif error.code == 429:
            category = "Достигнут лимит запросов Google"
        else:
            category = "Требуется дальнейшая диагностика Google API"
        return f"HTTP {error.code}; Google: {google_status}; {category}"
    except (OSError, UnicodeError, ValueError, TimeoutError):
        return "Сетевая ошибка или нераспознанный ответ Google"


async def gemini_check(update, context):
    """Only preapproved private tester may initiate this no-cost metadata call."""
    result = await asyncio.to_thread(probe_gemini_models)
    await update.effective_message.reply_text(
        "Проверка доступа к списку моделей Gemini (без генерации текста):\\n"
        + result
    )


def probe_gemini_generation():
    """Send one minimal nonpersonal POST; return only status and duration.

    No memory, private messages, or full Google response body are exposed.
    This distinguishes the GET model list from generateContent reliability.
    """
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview").strip()
    if not key:
        return "Нет GEMINI_API_KEY в Bothost"
    if not model or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for c in model):
        return "Некорректное имя GEMINI_MODEL"
    payload = json.dumps({
        "contents": [{"parts": [{"text": "Ответь одним словом: OK"}]}],
        "generationConfig": {"maxOutputTokens": 64}
    }).encode("utf-8")
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=payload,
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            data = json.loads(response.read(32768).decode("utf-8"))
        elapsed = round(time.monotonic() - start, 1)
        if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
            return f"HTTP 200, но неожиданный формат ответа; {elapsed} с"
        return f"HTTP 200: generateContent ответил; модель {model}; {elapsed} с"
    except urllib.error.HTTPError as error:
        status = "UNKNOWN"
        try:
            data = json.loads(error.read(4096).decode("utf-8"))
            info = data.get("error", {}) if isinstance(data, dict) else {}
            value = info.get("status") if isinstance(info, dict) else None
            if (isinstance(value, str) and 0 < len(value) <= 40
                    and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ_" for c in value)):
                status = value
        except (ValueError, UnicodeError, OSError):
            pass
        elapsed = round(time.monotonic() - start, 1)
        return f"HTTP {error.code}; Google: {status}; модель {model}; {elapsed} с"
    except (socket.timeout, TimeoutError):
        elapsed = round(time.monotonic() - start, 1)
        return f"Тайм-аут POST generateContent через {elapsed} с; модель {model}"
    except (urllib.error.URLError, OSError, UnicodeError, ValueError) as error:
        elapsed = round(time.monotonic() - start, 1)
        # Never interpolate exception strings: URLs could contain credentials.
        return f"Сетевая ошибка ({type(error).__name__}); {elapsed} с"


async def gemini_ping(update, context):
    """Owner-only diagnostic for minimal text-generation request."""
    result = await asyncio.to_thread(probe_gemini_generation)
    await update.effective_message.reply_text(
        "Минимальная проверка генерации Gemini (без памяти и истории):\n"
        + result
    )


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
        "network_check": network_check,
        "gemini_check": gemini_check,
        "gemini_ping": gemini_ping,
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
