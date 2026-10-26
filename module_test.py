
import json
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path("/app/data")
TEST_LOG = DATA_DIR / "module_test.jsonl"


def record_gemini_response():
    """Записывает контрольное событие после ответа Gemini."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "module": "module_test",
        "event": "gemini_response_received",
        "status": "success",
    }

    with TEST_LOG.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()

    return True
