"""Owner-only opt-in dialogue experiment, RAM history, no memory/database writes."""
import asyncio
import time
from tools.dialogue_explicit import classify, compose, safe_error, MODEL, PROTOCOL

KEY = 'explicit_live'

async def command(message, app, args, user_id, *, consent, gemini_key, gemini_model):
    if args == ['off']:
        app.bot_data.pop(KEY, None)
        await message.reply_text('Живая проверка выключена, её временная история удалена.')
        return
    if args == ['status']:
        state = app.bot_data.get(KEY)
        await message.reply_text('Живая проверка выключена.' if not state or state['owner'] != user_id else
            f"{PROTOCOL} · {state['provider']} · запросов {state['calls']}/40 · последняя категория: {state.get('last_kind','нет')}")
        return
    if args == ['reset']:
        state = app.bot_data.get(KEY)
        if state and state['owner'] == user_id:
            state['history'] = []
            state.pop('last_kind', None)
        await message.reply_text('Временный контекст очищен. Счётчик бюджета сохранён.')
        return
    if args not in (['gemini'], ['groq']):
        await message.reply_text('Формат: /dialogue_live gemini, groq, reset, status или off')
        return
    if not await consent():
        app.bot_data.pop(KEY, None)
        await message.reply_text('Для обработки сообщений требуется /memory_on.')
        return
    import os
    provider = args[0]
    if not (gemini_key if provider == 'gemini' else os.environ.get('GROQ_API_KEY','').strip()):
        await message.reply_text('Нет ключа выбранного провайдера.')
        return
    app.bot_data[KEY] = dict(owner=user_id, provider=provider, history=[], calls=0, ended=None)
    await message.reply_text(f'{PROTOCOL}: живая проверка включена ({provider}). Отправляй обычные сообщения. До 20 ответов, по два API-запроса на ответ; между запросами 15 секунд. История только в памяти процесса, без чтения и записи личной памяти. /dialogue_live reset — новый сюжет; /dialogue_live off — выход. После перезапуска режим выключен.')

async def handle(message, app, user_id, question, *, consent, gemini_key, gemini_model):
    state = app.bot_data.get(KEY)
    if not state or state['owner'] != user_id:
        return False
    if state['calls'] > 38 or len(state['history']) >= 40:
        await message.reply_text('Бюджет живой проверки исчерпан. /dialogue_live off — выход.')
        return True
    if len(question) > 6000 or sum(len(p['text']) for m in state['history'] for p in m['parts']) + len(question) > 24000:
        await message.reply_text('Контекст проверки заполнен или сообщение слишком длинное. /dialogue_live reset — новый сюжет.')
        return True
    options = dict(provider=state['provider'],gemini_key=gemini_key,gemini_model=gemini_model,groq_model=MODEL)
    async def call(fn, *args):
        if not await consent():
            raise PermissionError('Consent revoked')
        if state['ended'] is not None:
            await asyncio.sleep(max(0,15-(time.monotonic()-state['ended'])))
        if not await consent():
            raise PermissionError('Consent revoked')
        if state['calls'] >= 40:
            raise RuntimeError('Live request budget exhausted')
        state['calls'] += 1
        try:
            return await asyncio.to_thread(fn,*args,**options)
        finally:
            state['ended'] = time.monotonic()
    try:
        decision = await call(classify,question,state['history'])
        state['last_kind'] = decision['kind']
        answer = await call(compose,question,state['history'],decision)
        if not await consent():
            raise PermissionError('Consent revoked')
        # Deliver before retaining a successful turn; no primary private history writes.
        for offset in range(0,len(answer),4000):
            if not await consent():
                raise PermissionError('Consent revoked')
            await message.reply_text(answer[offset:offset+4000])
        state['history'].extend([dict(role='user',parts=[dict(text=question)]),dict(role='model',parts=[dict(text=answer)])])
    except PermissionError:
        app.bot_data.pop(KEY,None)
        await message.reply_text('Обработка остановлена: согласие отозвано; временная история удалена.')
    except Exception as error:
        import json
        await message.reply_text('Живая проверка остановила этот ответ: '+json.dumps(safe_error(error),ensure_ascii=False)+'. Автоматического повтора нет.')
    return True
