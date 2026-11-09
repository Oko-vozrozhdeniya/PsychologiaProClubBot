"""Native text-only control. No JSON response contract or production state."""
import asyncio
import json
import re
import time
import urllib.request
from memory.ai_providers import request_groq
from memory.ai_contract import ExtractionFormatError, ExtractionTruncatedError
from tools.dialogue_replies import MODEL, SHORT_POLICY, RULE, EXPECTATIONS
from tools.dialogue_lab import CASES, safe_error

INSTRUCTION = 'Ты доброжелательный собеседник. Общайся на русском языке.'+SHORT_POLICY+RULE
MAX_CALLS = 8


# Backward-compatible names for the plaintext-control tests and runner.
from memory.ai_providers import validate_plain_text as parse_text, request_plain_text


def generate_plain(provider, question, recent, *, gemini_key, gemini_model):
    return request_plain_text(provider,question,recent,instruction=INSTRUCTION,
        gemini_key=gemini_key,gemini_model=gemini_model,groq_model=MODEL)


async def compare_plain(emit, *, gemini_key, gemini_model):
    calls=0
    last_end=None
    lines=['Native plain-text control',
           'Та же короткая инструкция без FORMAT, JSON Schema и responseMimeType.',
           f'Gemini: {gemini_model}; Groq: {MODEL}.',
           'Переданы нативные сообщения с ролями. Предварительного режима и планировщика нет.',
           'У моделей различается реализация рассуждения; это не строгий рейтинг моделей.',
           'Оценивать содержание ответов, а не количество завершённых запросов.']
    for provider in ('gemini','groq'):
        for ident,q,recent,expected in CASES[:4]:
            lines.extend(['',f'{provider} / {ident}','Контекст: '+json.dumps(recent,ensure_ascii=False),
                          'Пользователь: '+q,'Критерий проверки: '+EXPECTATIONS[expected]])
            try:
                if calls>=MAX_CALLS:
                    raise RuntimeError('Text comparison budget exhausted')
                if last_end is not None:
                    await asyncio.sleep(max(0,15-(time.monotonic()-last_end)))
                calls+=1
                try:
                    reply=await asyncio.to_thread(generate_plain,provider,q,recent,
                        gemini_key=gemini_key,gemini_model=gemini_model)
                finally:
                    last_end=time.monotonic()
                lines.append('Ответ модели: '+reply)
                await emit(f'Обычный текст: {provider} / {ident}; {calls}/{MAX_CALLS} запросов.')
            except Exception as error:
                lines.append('ПРОВАЙДЕР ОСТАНОВЛЕН: '+json.dumps(safe_error(error),ensure_ascii=False))
                break
    lines.extend(['',f'Завершено: {calls}/{MAX_CALLS} запросов.',
                  'Ошибки останавливают соответствующего провайдера без повторов и подмены модели.',
                  'Прямой текст меняет и форматную инструкцию, и настройки декодирования относительно прежнего JSON-контроля.',
                  'Единичный успех не доказывает устойчивость или работу памяти. Рабочий диалог не переключён.'])
    await emit(f'Текстовый контроль завершён: {calls}/{MAX_CALLS} запросов. Полные ответы — в файле.')
    return '\n'.join(lines)
