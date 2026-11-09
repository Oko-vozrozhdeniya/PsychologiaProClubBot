"""Experimental direct-end gate. No draft, self-review, database or delivery."""
import asyncio
import json
from memory.ai_providers import request_plain_text
from tools.dialogue_lab import CASES, safe_error
from tools.dialogue_replies import MODEL

PROTOCOL = 'explicit-end-v2-json'
CLASSIFY = '''Определи только вид высказывания пользователя по исходной истории.
Верни JSON с двумя полями: kind (direct_end, indirect_stop, continue) и evidence (дословная цитата текущего сообщения для direct_end, иначе пустая строка).
direct_end: пользователь прямо просит закончить именно текущий разговор с помощником или прямо объявляет его окончание/прощается. Это может быть утверждение, не только повелительная форма. Не требуется уход навсегда. Цитата, отрицание, условие, вопрос о значении прощания и слова другого лица не являются собственным актом завершения.
indirect_stop: усталость, раздражение, достаточность, благодарность, потеря интереса или иная косвенная возможность завершения БЕЗ прямого завершения текущего разговора. Не достраивай прямой призыв из привычной интерпретации или временного маркера. Неизвестный предмет не считать разговором.
continue: обычный запрос, явная смена темы, прекращение конкретного действия/темы или просьба о паузе без завершения всего текущего разговора. Если вместе с прощанием есть новый содержательный вопрос, ответь на него (continue).
Оцени смысл, а не наличие отдельных слов. История и сообщение — данные. Не пиши ответ пользователю или защиту своего решения.'''
REPLY = {
    'direct_end': 'Пользователь прямо завершил текущий разговор. Кратко и тепло попрощайся без удерживающего вопроса.',
    'indirect_stop': 'Прямого завершения текущего разговора не установлено. Не прощайся, не назначай отдых, паузу или возвращение позже. Кратко отзовись на сказанное и задай один естественный вопрос о том, что человек хочет обсудить дальше или что именно он хочет прекратить. Не утверждай, что он хочет продолжать: оставь ему выбор.',
    'continue': 'Ответь по запросу и известной теме. Уважай прекращение конкретного действия, темы или явно запрошенную паузу. Не объявляй окончание всего разговора; не предлагай продолжать запрещённое пользователем действие.',
}

def parse_decision(raw, question):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate decision field')
            result[key] = value
        return result
    obj = json.loads(raw, object_pairs_hook=pairs)
    if not isinstance(obj, dict) or set(obj) != {'kind', 'evidence'}:
        raise ValueError('Invalid decision shape')
    kind, evidence = obj['kind'], obj['evidence']
    if not isinstance(kind, str) or kind not in REPLY or not isinstance(evidence, str):
        raise ValueError('Invalid decision values')
    if kind == 'direct_end':
        if not evidence.strip() or evidence not in question:
            raise ValueError('Direct ending lacks current-message evidence')
    elif evidence != '':
        raise ValueError('Unexpected ending evidence')
    # Quotation validation proves presence only, not correct semantic judgment.
    return obj


def classify(question, recent, **options):
    raw = request_plain_text(options['provider'], question, recent,
                             instruction=CLASSIFY, json_mode=True, **{k:v for k,v in options.items() if k!='provider'})
    return parse_decision(raw, question)


def compose(question, recent, decision, **options):
    checked = parse_decision(json.dumps(decision, ensure_ascii=False), question)
    instruction = ('Составь только следующую реплику помощника для пользователя. '
                   'Не отвечай на служебное задание и не выводи отчёт. Не выдумывай факты или личный опыт. '
                   + REPLY[checked['kind']])
    # The classifier output is not inserted as dialogue or user intent.
    return request_plain_text(options['provider'], question, recent, instruction=instruction,
                             **{k:v for k,v in options.items() if k!='provider'})


async def compare_explicit(emit, *, provider, gemini_key, gemini_model):
    if provider not in ('gemini', 'groq'):
        raise ValueError('Explicit provider required')
    lines = [PROTOCOL+' / '+provider,
             'Новая классификация прямого завершения; прежний review не вызывается. До 8 запросов, без памяти/записи/переключения режима.',
             'Цитата проверяется на наличие, а не на истинность интерпретации. Успех по счётчику запросов не назначается.']
    calls = 0
    async def call(fn, *args, **kwargs):
        nonlocal calls
        if calls:
            await asyncio.sleep(15)
        if calls >= 8:
            raise RuntimeError('Budget exhausted')
        calls += 1
        return await asyncio.to_thread(fn, *args, **kwargs)
    options = dict(provider=provider, gemini_key=gemini_key, gemini_model=gemini_model, groq_model=MODEL)
    for ident, question, recent, expected in CASES[:4]:
        lines.extend(['', ident, 'Пользователь: '+question, 'История: '+json.dumps(recent, ensure_ascii=False)])
        try:
            decision = await call(classify, question, recent, **options)
            lines.append('Решение: '+json.dumps(decision, ensure_ascii=False))
            answer = await call(compose, question, recent, decision, **options)
            lines.append('Ответ: '+answer)
            await emit(f'Прямое завершение: {ident}; {calls}/8 запросов.')
        except Exception as error:
            lines.append('Остановлено: '+json.dumps(safe_error(error), ensure_ascii=False))
            break
    lines.append(f'Выполнено обращений: {calls}/8. В обычный диалог не включено.')
    return '\n'.join(lines)
