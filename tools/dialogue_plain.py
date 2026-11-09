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


def parse_text(text, finish=None):
    if finish in ('length','MAX_TOKENS'):
        raise ExtractionTruncatedError('Text reply truncated')
    if not isinstance(text,str) or not text.strip() or len(text)>6000:
        raise ExtractionFormatError('Invalid text reply')
    return text.strip()


def generate_plain(provider, question, recent, *, gemini_key, gemini_model):
    if provider=='groq':
        return request_groq(question,recent,system_instruction=INSTRUCTION,
            response_task='plain',structured_parser=parse_text,model_override=MODEL,
            max_completion_tokens=4096,reasoning_effort='medium')
    if provider!='gemini':
        raise ValueError('Unknown text provider')
    if not gemini_key or not isinstance(gemini_model,str) or not re.fullmatch(r'[A-Za-z0-9._-]{1,100}',gemini_model):
        raise ValueError('Gemini text control unavailable')
    payload=dict(system_instruction=dict(parts=[dict(text=INSTRUCTION)]),
                 contents=[*recent,dict(role='user',parts=[dict(text=question)])],
                 generationConfig=dict(temperature=0,maxOutputTokens=4096))
    request=urllib.request.Request(
        'https://generativelanguage.googleapis.com/v1beta/models/'+gemini_model+':generateContent',
        data=json.dumps(payload,ensure_ascii=False).encode(),
        headers={'x-goog-api-key':gemini_key,'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(request,timeout=30) as response:
        result=json.loads(response.read(131072).decode())
    candidate=result['candidates'][0]
    # Thought parts are not conversational replies and must not enter the report.
    text=''.join(p.get('text','') for p in candidate['content']['parts']
                 if isinstance(p,dict) and not p.get('thought',False))
    return parse_text(text,candidate.get('finishReason'))


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
