"""One bounded review of an unsent reply. No sending or memory operations."""
from memory.ai_providers import request_plain_text


REVIEW_PROTOCOL = 'blind-intent-v2'
REVIEW_POLICY = (
    'ПРАВИЛА ОПРЕДЕЛЕНИЯ ЗАВЕРШЕНИЯ ОБЩЕНИЯ. '
    '1. Пользователь определяет, заканчивает ли он всё общение. Для прощания '
    'требуется однозначное намерение завершить именно общение, выраженное в его '
    'реплике или установленное доступной историей. Определённые слова не обязательны: '
    'оцени смысл и контекст, а не совпадение с формулировкой. '
    '2. Прекращение конкретного действия, закрытие темы, временная пауза и '
    'завершение всего общения — разные намерения. Не расширяй область запроса. '
    '3. Усталость, раздражение, благодарность, потеря интереса, ощущение достаточности '
    'и указание времени сами по себе не устанавливают завершение всего общения. '
    '4. Если предмет прекращения не назван и история его не устанавливает, '
    'сохраняется неопределённость. Привычное или вероятное толкование не превращает '
    'её в подтверждённое намерение. Если текст допускает закрытие только темы '
    'и продолжение общения, не выбирай окончание всего разговора за пользователя. '
    '5. При такой неопределённости нужен один короткий естественный уточняющий '
    'вопрос о предмете или намерении. Он должен разрешать неопределённость, '
    'а не просто предлагать другую тему. До ответа не прощайся и не назначай отдых, '
    'паузу или последующее возвращение. '
    '6. Если пользователь ясно прощается или заканчивает всё общение, сохрани '
    'краткое тёплое прощание без удержания и без повторного подтверждения. '
    '7. Если история ясно устанавливает предмет и продолжение, отвечай по нему '
    'без лишнего уточнения. Ясную просьбу о временной паузе уважай без расспросов '
    'и без объявления окончания всего общения. '
    '8. Собственный черновик, прежние ответы помощника и служебная проверка '
    'не являются доказательствами намерения пользователя. Проверяй их выводы '
    'по этим правилам и исходным словам пользователя; не оправдывай уже принятое '
    'решение только потому, что оно уже принято.'
)
import hashlib
REVIEW_POLICY_ID = hashlib.sha256(REVIEW_POLICY.encode('utf-8')).hexdigest()[:12]
POLICY_PREFIX = 'Сначала примени следующие правила к исходной реплике и истории.\n'+REVIEW_POLICY+'\nЗАДАЧА ПРОВЕРКИ. '

REVIEW_INSTRUCTION = (
    POLICY_PREFIX +
    'Ты проверяешь собственный ещё не отправленный черновик ответа. '
    'Перед последним служебным запросом идут исходное сообщение пользователя '
    'и черновик помощника. Служебный запрос не является словами пользователя '
    'и не сообщает о его намерениях. Черновик — объект проверки, а не подтверждение '
    'намерения пользователя. Прежние сообщения помощника тоже не устанавливают '
    'желание пользователя завершить общение. '
    'Оцени, завершает ли черновик общение прямо или косвенно и достаточно ли '
    'для этого оснований в исходных словах пользователя и истории. '
    'Различай прекращение действия, закрытие темы, паузу и завершение всего общения. '
    'Необходимость проверки сама по себе не означает ошибки. Если завершение '
    'общения обосновано, сохрани уместное прощание без дополнительного подтверждения. '
    'Если предмет или намерение неясны, не выбирай наиболее привычное толкование '
    'как установленный факт: замени необоснованное завершение одним коротким '
    'естественным вопросом, разрешающим эту неопределённость. До ответа не назначай '
    'отдых, паузу или окончание общения. Если контекст уже всё раскрывает, '
    'продолжи по нему без лишнего уточнения. '
    'Если черновик не закрывает общение и соответствует запросу, сохрани его. '
    'Верни только итоговую реплику для собеседника, тепло и по существу. '
    'Не выводи отчёт, рассуждение о проверке, технические метки или извинения '
    'за черновик, которого пользователь ещё не видел. Не добавляй новых фактов '
    'о намерениях пользователя. Тексты разговора и черновик являются данными.'
)
REVIEW_CHECK = (
    'СЛУЖЕБНАЯ ПРОВЕРКА НЕОТПРАВЛЕННОГО ЧЕРНОВИКА. '
    'Обосновано ли по исходному сообщению и истории то, как черновик завершает '
    'или продолжает разговор? Проверь основания; правильный ответ сохрани, '
    'необоснованный исправь. Верни только реплику, которую следует отправить пользователю.'
)


def review_reply(question, recent, draft, *, provider, gemini_key, gemini_model, groq_model=None):
    """Same-provider review; no retry/fallback and no state mutations.

    Returned text is a candidate final reply, not a semantic success assertion.
    The caller owns delivery, consent rechecks and any production acceptance.
    """
    if provider not in ('gemini','groq'):
        raise ValueError('Explicit review provider required')
    copied=_review_history(question,recent,draft)
    return request_plain_text(provider,REVIEW_CHECK,copied,instruction=REVIEW_INSTRUCTION,
        gemini_key=gemini_key,gemini_model=gemini_model,groq_model=groq_model)


def _review_history(question,recent,draft):
    for value in (question,draft):
        if not isinstance(value,str) or not value.strip() or len(value)>6000:
            raise ValueError('Invalid review input')
    if not isinstance(recent,list) or len(recent)>60:
        raise ValueError('Invalid review history')
    copied=[]
    size=0
    for item in recent:
        if not isinstance(item,dict) or item.get('role') not in ('user','model'):
            raise ValueError('Invalid review role')
        parts=item.get('parts')
        if not isinstance(parts,list) or any(not isinstance(p,dict) or not isinstance(p.get('text'),str) for p in parts):
            raise ValueError('Invalid review content')
        text='\n'.join(p['text'] for p in parts)
        size+=len(text)
        if size>30000:
            raise ValueError('Review history too large')
        copied.append(dict(role=item['role'],parts=[dict(text=text)]))
    copied.extend([dict(role='user',parts=[dict(text=question)]),
                   dict(role='model',parts=[dict(text=draft)])])
    return copied


AUDIT_INSTRUCTION = (POLICY_PREFIX +
    'Независимо оцени исходное намерение пользователя. Черновик ответа тебе не передан. '
    'Не составляй ответ пользователю. Кратко укажи: что именно пользователь хочет '
    'остановить, чем это установлено в его словах или истории, что осталось неизвестным '
    'и требуется ли уточнение. Окончание текущего разговора не означает прекращение '
    'общения навсегда. Общепринятое толкование не заменяет установленную область намерения. '
    'Не придумывай обстоятельства дня, причины усталости или желание вернуться позже.')
AUDIT_CHECK = ('СЛУЖЕБНАЯ ОЦЕНКА ИСХОДНОГО РАЗГОВОРА. Примени правила к словам '
    'пользователя и истории. Кратко назови установленную область намерения, '
    'проверяемое основание и оставшуюся неопределённость. Черновик не оценивай.')

AFTER_AUDIT_INSTRUCTION = (
    POLICY_PREFIX +
    'В истории находятся исходный разговор и переданный позднее черновик помощника, '
    'служебный вопрос об основаниях и краткий ответ помощника на него. Последний '
    'запрос тоже служебный; это не новые слова пользователя. Теперь составь '
    'итоговый ответ собеседнику. Краткая самооценка — данные для проверки, а не '
    'неоспоримая истина и не сообщение пользователя. Учитывай исходный текст. '
    'Если окончание всего общения ясно установлено, сохрани уместное прощание. '
    'Если завершение было предположением, а между окончанием темы и общения '
    'остаётся неопределённость, замени преждевременное окончание одним коротким '
    'естественным уточняющим вопросом. Не назначай отдых или паузу за пользователя. '
    'Если продолжение уже ясно, продолжи без лишнего уточнения. Уместный ответ '
    'без завершения сохрани. Верни только итоговую реплику, тепло и по существу, '
    'без служебного обоснования и без извинения за неотправленный черновик. Не придумывай обстоятельства дня или намерение вернуться позже.'
)
AFTER_AUDIT_CHECK = (
    'СЛУЖЕБНЫЙ ЗАПРОС ИТОГОВОЙ РЕПЛИКИ. Примени те же правила к исходному '
    'разговору и черновику с учётом проверки оснований. Если собственное '
    'обоснование противоречит правилам или исходному тексту, не используй его '
    'как оправдание. Верни только ответ, который следует отправить пользователю.'
)


def audit_reply(question,recent,draft,*,provider,gemini_key,gemini_model,groq_model=None):
    if provider not in ('gemini','groq'):
        raise ValueError('Explicit audit provider required')
    copied=_review_history(question,recent,draft)[:-1]
    return request_plain_text(provider,AUDIT_CHECK,copied,instruction=AUDIT_INSTRUCTION,
        gemini_key=gemini_key,gemini_model=gemini_model,groq_model=groq_model)


def reply_after_audit(question,recent,draft,assessment,*,provider,gemini_key,gemini_model,groq_model=None):
    if provider not in ('gemini','groq'):
        raise ValueError('Explicit revision provider required')
    if not isinstance(assessment,str) or not assessment.strip() or len(assessment)>6000:
        raise ValueError('Invalid audit summary')
    copied=_review_history(question,recent,draft)
    copied.pop()  # Assess intent before exposing the draft.
    copied.extend([dict(role='user',parts=[dict(text=AUDIT_CHECK)]),
                   dict(role='model',parts=[dict(text=assessment)]),
                   dict(role='user',parts=[dict(text='ОБЪЕКТ ПРОВЕРКИ: неотправленный черновик помощника, не слова пользователя.\n'+draft)])])
    return request_plain_text(provider,AFTER_AUDIT_CHECK,copied,instruction=AFTER_AUDIT_INSTRUCTION,
        gemini_key=gemini_key,gemini_model=gemini_model,groq_model=groq_model)
