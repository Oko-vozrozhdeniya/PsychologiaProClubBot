"""Text-only Groq backup for the authorized private AI conversation.

The privacy and identity gate lives in bot.private_message. This module gets
only the specific user's authorized history/context supplied by that gate.
The original Gemini archive-analysis mode is not modified.
"""
import json
import os
import re
import socket
import urllib.error
import urllib.request

from memory.gemini_prompt_adapter import append_private_memory
from memory.ai_contract import instructions_for, parse_extraction, ExtractionFormatError
from memory.ai_adapters import extraction_settings, search_settings

GROQ_DEFAULT_MODEL = "openai/gpt-oss-20b"


class GroqRequestError(Exception):
    def __init__(self, error_type="general", http_code=None, reason="Ошибка Groq",
                 request_stats=None):
        super().__init__(error_type)
        self.error_type = error_type
        self.http_code = http_code
        self.reason = reason
        self.request_stats = request_stats or {}


def _describe_groq_http_error(error):
    """Safely classify provider failure without exposing raw bodies or secrets.

    Only allowlisted-shaped machine codes and semantic classifications are
    reported. Never forward remote error messages into Telegram or log files.
    """
    try:
        raw = error.read(4096)
        parsed = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, UnicodeError, TypeError, AttributeError):
        return "Ответ об ошибке не является JSON; возможна блокировка на сетевом уровне"
    data = parsed.get("error") if isinstance(parsed, dict) else None
    if not isinstance(data, dict):
        return "Ошибка JSON без стандартного поля Groq error"
    err_type = data.get("type")
    err_code = data.get("code")
    message = data.get("message")
    safe_type = (err_type if isinstance(err_type, str) and
                 re.fullmatch(r"[a-z0-9_]{1,64}", err_type)
                 else "не указан")
    safe_code = (err_code if isinstance(err_code, str) and
                 re.fullmatch(r"[a-z0-9_]{1,80}", err_code)
                 else "не указан")
    description = "Причина требует дополнительной проверки"
    raw_msg = message.lower() if isinstance(message, str) else ""
    if safe_code == "model_permission_blocked_org":
        description = "Модель заблокирована на уровне организации"
    elif safe_code == "model_permission_blocked_project":
        description = "Модель заблокирована на уровне проекта"
    elif (("region" in raw_msg or "location" in raw_msg or "country" in raw_msg)
          and ("not supported" in raw_msg or "restricted" in raw_msg
               or "blocked" in raw_msg)):
        description = "Сервис отклоняет регион или сетевой адрес запроса"
    elif "signup_blocked" in raw_msg or safe_code == "signup_blocked":
        description = "Ограничение регистрации или аккаунта"
    elif "organization" in raw_msg and (
        "restricted" in raw_msg or "blocked" in raw_msg
    ):
        description = "Ограничение организации Groq"
    return f"тип={safe_type}; код={safe_code}; {description}"


def request_groq(question, history=None, personal_memory_context=None, *,
                 system_instruction=None, response_task="extraction", reply_mode="answer", structured_parser=None):
    """Make one explicit, bounded Groq chat request. No secret/body logging."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    model = os.environ.get("GROQ_MODEL", GROQ_DEFAULT_MODEL).strip()
    if not key:
        raise GroqRequestError(reason="Отсутствует GROQ_API_KEY")
    if not model or len(model) > 80 or any(
        c not in "abcdefghijklmnopqrstuvwxyz0123456789._-/" for c in model
    ):
        raise GroqRequestError(reason="Некорректная модель Groq")
    if not isinstance(question, str) or not question.strip():
        raise GroqRequestError(reason="Пустой запрос")
    if system_instruction is not None and (
        not isinstance(system_instruction, str) or not system_instruction.strip()
    ):
        raise GroqRequestError(reason="Некорректная инструкция запроса")
    if personal_memory_context:
        question = append_private_memory(question, personal_memory_context)
    instructions = system_instruction if system_instruction is not None else instructions_for("private", reply_mode=reply_mode)
    messages = [{"role": "system", "content": instructions}]
    if history is not None and not isinstance(history, (tuple, list)):
        raise GroqRequestError(reason="Некорректный формат истории")
    for item in (history or []):
        if not isinstance(item, dict) or item.get("role") not in ("user", "model"):
            continue
        content = item.get("parts", [])
        text = "\n".join(
            part["text"] for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
        if text:
            messages.append({"role": ("assistant" if item["role"] == "model" else "user"),
                             "content": text})
    messages.append({"role": "user", "content": question})
    payload = json.dumps({
        "model": model, "messages": messages, "max_completion_tokens": 1200,
        "stream": False,
        **((search_settings("Groq")
            if response_task in ("search","closure_review") else extraction_settings("Groq"))
           if system_instruction is not None else {}),
    }, ensure_ascii=False).encode("utf-8")
    stats = {
        "request_bytes": len(payload),
        "history_messages": len(messages) - 2,
        "history_chars": sum(len(m["content"]) for m in messages[1:-1]),
        "request_mode": "личный диалог",
    }
    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json",
                 "User-Agent": "PsychologiaProClubBot/1.0"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read(131072).decode("utf-8"))
        result_choice = result["choices"][0]
        choice = result_choice["message"]["content"]
        if system_instruction is not None:
            if structured_parser is not None:
                structured_parser(choice, result_choice.get("finish_reason"))
            elif response_task == "search":
                from memory.ai_contract import parse_search_plan
                parse_search_plan(choice, result_choice.get("finish_reason"))
            elif response_task == "closure_review":
                from memory.ai_contract import parse_closure_review
                parse_closure_review(choice,result_choice.get("finish_reason"),user_messages=[])
            elif response_task == "extraction":
                parse_extraction(choice, result_choice.get("finish_reason"))
            else:
                raise ValueError("Unknown structured task")
        if not isinstance(choice, str) or not choice.strip():
            raise ValueError("No text returned")
        return choice.strip()
    except urllib.error.HTTPError as error:
        code = error.code
        category = ("rate_limit" if code == 429 else
                    "unavailable" if code in (500, 502, 503, 504) else
                    "general")
        reason = f"Groq HTTP {code}; {_describe_groq_http_error(error)}"
        raise GroqRequestError(
            category, http_code=code, reason=reason,
            request_stats=stats
        ) from error
    except (socket.timeout, TimeoutError) as error:
        raise GroqRequestError(
            "timeout", reason="Превышено время ожидания Groq",
            request_stats=stats
        ) from error
    except urllib.error.URLError as error:
        raise GroqRequestError(
            "unavailable", reason="Соединение с Groq недоступно",
            request_stats=stats
        ) from error
    except ExtractionFormatError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
        raise GroqRequestError(
            "general", reason="Не удалось прочитать ответ Groq",
            request_stats=stats
        ) from error


def request_personal_ai(question, history, personal_memory_context, mode,
                        gemini_call, *, reply_mode="answer"):
    """Generate freely under shared rules; return the actual provider."""
    instructions_for("private", reply_mode=reply_mode)  # Validate before network use.
    reply_options = {"reply_mode": reply_mode} if reply_mode != "answer" else {}
    if mode == "groq":
        return request_groq(question, history, personal_memory_context, **reply_options), "Groq"
    if mode == "auto":
        try:
            answer = gemini_call(
                question, None, history, None, personal_memory_context, 15, **reply_options
            )
            return answer, "Gemini"
        except Exception as error:
            # GeminiRequestError is defined in bot.py; avoid circular imports.
            if getattr(error, "error_type", None) not in (
                "timeout", "unavailable", "rate_limit"
            ):
                raise
            return request_groq(question, history, personal_memory_context, **reply_options), "Groq"
    if mode != "gemini":
        raise ValueError("Unknown test AI mode")
    answer = gemini_call(question, None, history, None, personal_memory_context, **reply_options)
    return answer, "Gemini"


def request_memory_search(question, history, active_context, mode, *, gemini_key,
                          gemini_model):
    """One bounded semantic query plan; no full archive or permissions from AI."""
    from memory.ai_contract import parse_search_plan
    from memory.ai_adapters import search_settings
    if mode not in ('auto','gemini','groq'):
        raise ValueError('Invalid AI mode')
    recent, size = [], 0
    for item in reversed((history or [])[-8:]):
        if not isinstance(item,dict) or item.get('role') not in ('user','model'):
            continue
        value='\n'.join(p['text'] for p in item.get('parts',[]) if isinstance(p,dict)
                         and isinstance(p.get('text'),str))[:1800]
        if size+len(value)>6000:
            break
        recent.insert(0,{'role':item['role'],'text':value})
        size+=len(value)
    prompt=json.dumps({'question':question[:6000], 'recent':recent,
                       'active_context':active_context[:6000]},ensure_ascii=False)
    instruction=instructions_for('search')
    user_messages=[question]+[item['text'] for item in recent if item['role']=='user']
    return _request_semantic_json(prompt,instruction,mode,
        lambda text,finish=None: parse_search_plan(text,finish,user_messages=user_messages),
        gemini_key=gemini_key,gemini_model=gemini_model)


def request_closure_review(question,history,mode,*,gemini_key,gemini_model):
    """Independent scope check, only for proposed farewells; no first verdict supplied."""
    from memory.ai_contract import CLOSURE_REVIEW_INSTRUCTIONS, parse_closure_review
    recent=[];size=0
    for item in reversed((history or [])[-60:]):
        if not isinstance(item,dict) or item.get('role') not in ('user','model'):
            continue
        value='\n'.join(p['text'] for p in item.get('parts',[]) if isinstance(p,dict)
            and isinstance(p.get('text'),str))[:1800]
        if size+len(value)>16000:
            break
        recent.insert(0,{'role':item['role'],'text':value});size+=len(value)
    prompt=json.dumps({'question':question[:6000],'recent':recent},ensure_ascii=False)
    user_messages=[question[:6000]]+[i['text'] for i in recent if i['role']=='user']
    return _request_semantic_json(prompt,CLOSURE_REVIEW_INSTRUCTIONS,mode,
        lambda text,finish=None: parse_closure_review(text,finish,user_messages=user_messages),
        gemini_key=gemini_key,gemini_model=gemini_model,response_task="closure_review")


def _request_semantic_json(prompt,instruction,mode,parser,*,gemini_key,gemini_model,response_task="search"):
    """Shared native structured transport for bounded semantic tasks."""
    if mode not in ('auto','gemini','groq'):
        raise ValueError('Invalid AI mode')
    def groq():
        text=request_groq(prompt,system_instruction=instruction,response_task=response_task,structured_parser=parser)
        return parser(text),'Groq'
    if mode=='groq':
        return groq()
    try:
        if not gemini_key or not isinstance(gemini_model,str) or not re.fullmatch(r'[A-Za-z0-9._-]{1,100}',gemini_model):
            raise ValueError('Gemini search unavailable')
        payload={'system_instruction':{'parts':[{'text':instruction}]},
                 'contents':[{'role':'user','parts':[{'text':prompt}]}],
                 'generationConfig':search_settings('Gemini')}
        request=urllib.request.Request(
            'https://generativelanguage.googleapis.com/v1beta/models/'+gemini_model+':generateContent',
            data=json.dumps(payload,ensure_ascii=False).encode(),
            headers={'x-goog-api-key':gemini_key,'Content-Type':'application/json'},method='POST')
        with urllib.request.urlopen(request,timeout=15) as response:
            data=json.loads(response.read(131072).decode())
        candidate=data['candidates'][0]
        text=''.join(p.get('text','') for p in candidate['content']['parts'] if isinstance(p,dict))
        return parser(text,candidate.get('finishReason')),'Gemini'
    except Exception:
        if mode!='auto':
            raise
        return groq()
