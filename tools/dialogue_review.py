"""Review four existing synthetic drafts from the owner's dialogue-plain report."""
import asyncio
import json
import time
from memory.reply_review import review_reply
from tools.dialogue_lab import CASES,safe_error
from tools.dialogue_replies import MODEL,EXPECTATIONS

# Exact existing outputs from the uploaded dialogue-plain.txt, not response
# templates. These fixtures are NEVER inserted into the review instructions.
DRAFTS = {
    'gemini': (
        'Понимаю, отдыхайте. Всего доброго!',
        'Хорошего вам вечера!',
        'До свидания! Всего вам доброго.',
        'Понимаю, цифры и отчеты могут сильно утомлять. Давай переключимся на что-то приятное — куда бы ты хотел отправиться, если бы мог прямо сейчас оказаться в любой точке мира?',
    ),
    'groq': (
        'Спасибо, до встречи!',
        'Приятного вам дня!',
        'До свидания! Всего доброго.',
        'Отлично! Куда бы ты хотел отправиться в следующем путешествии?',
    ),
}


async def compare_reviews(emit, *, provider, gemini_key, gemini_model):
    if provider not in DRAFTS:
        raise ValueError('Invalid review profile')
    calls=0
    last_end=None
    lines=[f'Unsent draft review / {provider} / '+(gemini_model if provider=='gemini' else MODEL),
           'Черновики взяты из уже полученного dialogue-plain.txt. Новые первичные ответы не генерировались.',
           'Один дополнительный вызов на черновик: исходный разговор, собственный черновик, нейтральный служебный запрос.',
           'Проверяются все четыре черновика, включая обоснованное прощание и продолжение темы.',
           'Служебный запрос не выдаётся за новое сообщение пользователя. Нет автоматической оценки успеха.']
    for case,draft in zip(CASES[:4],DRAFTS[provider]):
        ident,q,recent,expected=case
        lines.extend(['',ident,'Контекст: '+json.dumps(recent,ensure_ascii=False),
                      'Пользователь: '+q,'Неотправленный черновик: '+draft,
                      'Критерий разбора: '+EXPECTATIONS[expected]])
        try:
            if calls>=4:
                raise RuntimeError('Draft review budget exhausted')
            if last_end is not None:
                await asyncio.sleep(max(0,15-(time.monotonic()-last_end)))
            calls+=1
            try:
                reply=await asyncio.to_thread(review_reply,q,recent,draft,provider=provider,
                    gemini_key=gemini_key,gemini_model=gemini_model,groq_model=MODEL)
            finally:
                last_end=time.monotonic()
            lines.append('Ответ после проверки: '+reply)
            await emit(f'Проверка черновиков: {ident}; {calls}/4 запросов.')
        except Exception as error:
            lines.append('ПРОГОН ОСТАНОВЛЕН: '+json.dumps(safe_error(error),ensure_ascii=False))
            break
    lines.extend(['',f'Завершено: {calls}/4 запросов.',
                  'Успех требует исправления необоснованных завершений и сохранения обоснованных ответов.',
                  'Это первый контроль на известных черновиках, не проверка устойчивости новых формулировок.',
                  'В рабочий диалог проверка ещё не включена. История, память и настройки не изменены.'])
    await emit(f'Проверка черновиков завершена: {calls}/4 запросов. До и после — в файле.')
    return '\n'.join(lines)
