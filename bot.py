
import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

logging.basicConfig(level=logging.INFO)

TOKEN = os.environ["BOT_TOKEN"]

DATA_DIR = Path("/app/data")
TEXT_FILE = DATA_DIR / "messages.txt"
JSON_FILE = DATA_DIR / "messages.jsonl"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Здравствуйте! Я помощник клуба «Психология Pro». "
        "Подключение работает успешно!\n\n"
        "Команда /analyze проверяет архив сообщений."
    )


async def save_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    chat = update.effective_chat

    if not message or not chat or chat.type not in ("group", "supergroup"):
        return

    if not message.text:
        return

    timestamp = datetime.now(timezone.utc).isoformat()
    topic_id = message.message_thread_id or 0
    sender_id = message.from_user.id if message.from_user else 0

    record = {
        "timestamp": timestamp,
        "chat_id": chat.id,
        "topic_id": topic_id,
        "user_id": sender_id,
        "message_id": message.id,
        "text": message.text,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with TEXT_FILE.open("a", encoding="utf-8") as file:
        file.write(
            f"{timestamp} | chat={chat.id} | topic={topic_id} "
            f"| user={sender_id} | message={message.id}\n"
            f"{message.text}\n\n"
        )

    with JSON_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    logging.info("GROUP_MESSAGE_SAVED message_id=%s", message.id)


async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not JSON_FILE.exists():
        await update.message.reply_text(
            "Архив сообщений пока не найден."
        )
        return

    records = []

    try:
        with JSON_FILE.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        logging.exception("Archive reading failed")
        await update.message.reply_text(
            "Не удалось прочитать архив сообщений."
        )
        return

    if not records:
        await update.message.reply_text(
            "Архив существует, но сообщений пока нет."
        )
        return

    last_message = records[-1]

    await update.message.reply_text(
        "Проверка архива успешна!\n\n"
        f"Всего сохранённых сообщений: {len(records)}\n"
        f"Последнее сообщение:\n{last_message['text']}\n\n"
        "Следующий этап: подключение искусственного интеллекта."
    )


def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", analyze))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, save_message)
    )

    print("Бот «Психология Pro» запущен", flush=True)
    app.run_polling()


if __name__ == "__main__":
    main()
