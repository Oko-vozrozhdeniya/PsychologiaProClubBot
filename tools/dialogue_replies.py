"""Synthetic natural-response comparison; no production chat, DB or mode writes."""
import asyncio
import json
import time
from memory.ai_contract import instructions_for, ExtractionFormatError, ExtractionTruncatedError
from memory.ai_providers import request_groq, request_memory_search, request_closure_review
from tools.dialogue_lab import CASES, history, safe_error

MODEL = 'openai/gpt-oss-120b'
MAX_CALLS = 28
RULE = (
    ' При неопределённости между прекращением действия, темы и всего общения '
    'задай один короткий естественный уточняющий вопрос и дождись ответа. '
    'До уточнения не объявляй разговор законченным, не назначай отдых или паузу. '
    'Если контекст уже устанавливает намерение, ответь по нему без лишнего вопроса. '
    'Уважай ясное прощание. Ответ на твой уточняющий вопрос прочитай в связи с этим '
    'вопросом, а не изолированно. Не копируй готовую формулировку; говори тепло '
    'и кратко по смыслу разговора.'
)
FORMAT = ' Верни JSON только с полем reply — текстом твоего ответа собеседнику. Не классифицируй намерение и не объясняй выбор ответа.'
SCHEMA = dict(type='object',properties=dict(reply=dict(type='string')),required=['reply'],additionalProperties=False)
EXPECTATIONS = {
    'answer':'Продолжить по понятному контексту, без лишнего уточнения и прощания.',
    'clarify':'Коротко уточнить неизвестный предмет или намерение; не объявлять завершение.',
    'close':'Коротко и тепло попрощаться, без удерживающего вопроса.',
}


def parse_reply(text, finish=None):
    if finish in ('length','MAX_TOKENS'):
        raise ExtractionTruncatedError('Reply truncated')
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:
                raise ExtractionFormatError('Duplicate reply field')
            d[k]=v
        return d
    try:
        if not isinstance(text,str) or len(text)>16000:
            raise ValueError
        d=json.loads(text,object_pairs_hook=pairs)
        if (not isinstance(d,dict) or set(d)!={'reply'} or not isinstance(d['reply'],str)
                or not d['reply'].strip() or len(d['reply'])>2000):
            raise ValueError
        return d['reply'].strip()
    except (ValueError,TypeError,RecursionError) as e:
        raise ExtractionFormatError('Invalid reply envelope') from e


def generate_reply(question, recent, mode):
    # Same model, rules, JSON wrapper and generation settings for both arms.
    # The wrapper carries natural text only; there is no dialogue-action parser.
    text=request_groq(question,recent,system_instruction=instructions_for('private',reply_mode=mode)+RULE+FORMAT,
        structured_parser=parse_reply,response_schema=SCHEMA,model_override=MODEL,
        max_completion_tokens=4096,reasoning_effort='medium',response_task='search')
    return parse_reply(text)


async def compare_replies(emit):
    """Return a complete text report. No automatic semantic pass/fail judgement."""
    calls=0
    last_end=None
    lines=[f'Natural replies / generator {MODEL}',
           'Контролируемое сравнение: генератор и параметры одинаковы; базовый планировщик использует настроенную модель Groq.',
           'Это не полный рабочий путь: личная память, архив, согласия и запись фактов не проверяются.',
           'JSON проверяет только форму. Семантической автоматической оценки нет.']
    first_direct=None

    async def request(fn,*args,**kwargs):
        nonlocal calls,last_end
        if calls>=MAX_CALLS:
            raise RuntimeError('Reply comparison call budget exhausted')
        if last_end is not None:
            await asyncio.sleep(max(0,6-(time.monotonic()-last_end)))
        calls+=1
        try:
            return await asyncio.to_thread(fn,*args,**kwargs)
        finally:
            last_end=time.monotonic()

    async def case_reply(case, arm, stage):
        nonlocal first_direct
        ident,q,recent,expected=case
        mode='answer'
        trace=[]
        lines.extend(['',f'{stage} / {ident} / {arm}',
                      'Контекст: '+json.dumps(recent,ensure_ascii=False),
                      'Пользователь: '+q,'Критерий проверки: '+EXPECTATIONS[expected]])
        if arm=='baseline':
            plan,_=await request(request_memory_search,q,recent,'','groq',gemini_key='',gemini_model='')
            mode=plan['reply_mode']
            trace.append('Исходный режим: '+mode)
            if mode=='brief_close':
                target,_=await request(request_closure_review,q,recent,'groq',gemini_key='',gemini_model='')
                trace.append('Проверка завершения: '+target)
                mode='brief_close' if target=='conversation' else 'brief_clarify' if target=='unclear' else 'answer'
        reply=await request(generate_reply,q,recent,mode)
        if ident==CASES[0][0] and arm=='direct' and stage=='основные':
            first_direct=reply
        lines.extend([*trace,'Ответ модели: '+reply])
        await emit(f'{stage}: {ident} / {arm}; запросов {calls}/{MAX_CALLS}. Ответ сохранён в отчёт.')

    try:
        # Baseline up to three calls/case; direct exactly one: at most 16 here.
        for case in CASES[:4]:
            for arm in ('baseline','direct'):
                await case_reply(case,arm,'основные')
        # Independent repetitions, same four input conversations, no reused answer.
        for case in CASES[:4]:
            await case_reply(case,'direct','повтор')
        for case in CASES[4:]:
            await case_reply(case,'direct','дополнительные')
        # Branch the actual generated response, rather than assuming it clarified.
        # Review must first establish whether that response asked a clarification.
        for ident,q,expected in (
                ('followup_topic','Только эту тему. Давай поговорим о путешествиях.','answer'),
                ('followup_end','Да, хочу закончить разговор. До свидания.','close')):
            await case_reply((ident,q,history((CASES[0][1],first_direct)),expected),'direct','ответ на уточнение')
    except Exception as error:
        lines.extend(['','ПРОГОН ОСТАНОВЛЕН: '+json.dumps(safe_error(error),ensure_ascii=False)])
        await emit('Прогон остановлен: '+json.dumps(safe_error(error),ensure_ascii=False))
    lines.extend(['',f'Завершено: {calls}/{MAX_CALLS} запросов.',
                  'Ответы требуют содержательного разбора. Наличие отчёта не означает успешную приёмку.',
                  'Ветви продолжения проверяют именно уточнение только если предыдущий ответ действительно его содержал.',
                  'Рабочий диалог и модель не переключены.'])
    await emit(f'Сравнение завершено: {calls}/{MAX_CALLS} запросов. Полные ответы — в файле; автоматической оценки понимания нет.')
    return '\n'.join(lines)
