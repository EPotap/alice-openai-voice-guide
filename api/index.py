"""HTTPS-вебхук приватного навыка Алисы для Vercel.

Зависимостей, кроме стандартной библиотеки Python, нет. Все персональные
настройки передаются через переменные окружения Vercel.
"""

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request


API_URL = "https://api.anthropic.com/v1/messages"
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
MAX_ANSWER_CHARS = 900  # Лимит поля response.text у Алисы: 1024.
CLAUDE_TIMEOUT_SECONDS = 3.6  # У Яндекс Диалогов всего 4,5 с, включая сеть.
MAX_CONTEXT_BYTES = 850  # session_state Яндекс Диалогов ограничен 1 КБ.

INSTRUCTIONS = (
    "Ты голосовой помощник в приватном навыке Алисы и работаешь через Claude API Anthropic. "
    "Если пользователь спрашивает, на какой модели или технологии ты работаешь, честно скажи: "
    "«Я работаю на Claude через API Anthropic». Никогда не утверждай, что работаешь на YandexGPT "
    "или другой модели. Отвечай на русском языке "
    "естественно и кратко: обычно одно-два предложения, до 40 слов. Пиши только текст для "
    "произнесения вслух, без Markdown, списков, ссылок и эмодзи. Если данных "
    "недостаточно, честно скажи об этом. Учитывай сказанное в этой беседе, "
    "но не обещай помнить его после выхода из навыка. Не заявляй, что выполнил "
    "действие во внешнем сервисе, если оно не было выполнено."
)
SEARCH_PROMPT = (
    "Для этого вопроса обязательно используй доступный веб-поиск. Ответь на русском кратко, "
    "только по найденным данным. Назови источник обычными словами. "
    "Не произноси адреса сайтов. Если подтверждения нет, скажи об этом."
)


class ClaudeError(Exception):
    """Ошибка обращения к Claude API."""


def _api_request(key, payload, timeout=CLAUDE_TIMEOUT_SECONDS):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload else None
    headers = {
        "Authorization": "Bearer " + key,
        "anthropic-version": "2026-10-21",
        "Content-Type": "application/json",
    }
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
    if workspace_id:
        headers["anthropic-workspace-id"] = workspace_id
    req = urllib.request.Request(
        API_URL,
        data=body,
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return json.load(res)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
            error = detail.get("error", {})
            logging.warning(
                "Claude API HTTP %s: %s — %s",
                exc.code,
                str(error.get("code") or error.get("type") or "unknown")[:80],
                str(error.get("message") or "")[:300],
            )
        except (OSError, ValueError, AttributeError):
            logging.warning("Claude API HTTP %s", exc.code)
        if exc.code == 401:
            raise ClaudeError("Ключ Claude API не принят.") from exc
        if exc.code == 400:
            raise ClaudeError("Запрос к Claude API отклонён. Проверьте модель и настройки.") from exc
        if exc.code == 429:
            raise ClaudeError("Сейчас достигнут лимит Claude API.") from exc
        raise ClaudeError("Сервис ответов временно недоступен.") from exc
    except (OSError, ValueError) as exc:
        logging.warning("Claude API request failed: %s", type(exc).__name__)
        raise ClaudeError("Не удалось связаться с сервисом ответов.") from exc


def _needs_web_search(command):
    if re.search(r"\b(найди|поищи|ищи|загугли|погугли)\b", command):
        return True
    if re.search(r"\b(в интернете|в сети|по интернету)\b", command):
        return True
    if re.search(r"\b(новости|новостей|погода|погоду|прогноз погоды)\b", command):
        return True
    return command.startswith(("что нового", "что произошло сегодня", "какой сейчас курс"))


def _answer_text(data):
    parts = []
    for part in data.get("content", []):
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            parts.append(part["text"])
    text = " ".join(parts)
    text = re.sub(r"【[^】]{1,160}】|\ue200cite\ue202.*?\ue201", "", text)
    text = re.sub(r"\[([^]]+)\]\(https?://[^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s*\((?:[\w-]+\.)+[\w-]{2,}(?:/[^)]*)?\)", "", text)
    text = " ".join(text.split()).strip()
    if len(text) > MAX_ANSWER_CHARS:
        cut = text[:MAX_ANSWER_CHARS]
        sentence_end = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
        text = cut[: sentence_end + 1] if sentence_end > 400 else cut.rstrip() + "…"
    return text or "Ответ пока не получился. Попробуйте спросить иначе."


def _source_buttons(data):
    buttons = []
    seen = set()
    for part in data.get("content", []):
        if part.get("type") != "text":
            continue
        for citation in part.get("citations", []):
            url = citation.get("url", "")
            title = " ".join(str(citation.get("title") or "Источник").split())
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme != "https" or not parsed.netloc or len(url.encode("utf-8")) > 1024 or url in seen:
                continue
            seen.add(url)
            buttons.append({"title": ("Источник: " + title)[:64], "url": url, "hide": True})
            if len(buttons) == 3:
                return buttons
    return buttons


def _say(text, state=None, end_session=False):
    result = {
        "version": "1.0",
        "response": {"text": text[:1024], "end_session": end_session},
    }
    if state is not None and not end_session:
        result["session_state"] = state
    return result


def _conversation_state(question, answer):
    """Хранит короткий контекст, не превышая лимит session_state Яндекса."""
    state = {"last_question": question[:280], "last_answer": answer[:360]}
    while len(json.dumps(state, ensure_ascii=False).encode("utf-8")) > MAX_CONTEXT_BYTES:
        if len(state["last_answer"]) > 80:
            state["last_answer"] = state["last_answer"][:-40]
        elif len(state["last_question"]) > 80:
            state["last_question"] = state["last_question"][:-40]
        else:
            return {}
    return state


def _finish(data, question):
    answer = _answer_text(data)
    result = _say(answer, _conversation_state(question, answer))
    buttons = _source_buttons(data)
    if buttons:
        result["response"]["buttons"] = buttons
    return result


def handler_alice(event, context):
    """Обработчик события Яндекс Диалогов."""
    del context
    if not isinstance(event, dict):
        return _say("Не удалось прочитать запрос. Повторите, пожалуйста.")

    req = event.get("request") or {}
    utterance = (req.get("command") or req.get("original_utterance") or "").strip()
    command = utterance.casefold().strip(" .,!?;:")
    state = (event.get("state") or {}).get("session") or {}
    if not isinstance(state, dict):
        state = {}
    if command in ("хватит", "выход", "закончить", "стоп"):
        return _say("До встречи!", end_session=True)
    if command in ("сброс", "забудь разговор", "начать заново"):
        return _say("Начинаем новый разговор. Что хотите спросить?", {})
    if re.search(r"\b(поставь|установи|заведи|включи|создай)\b.*\b(будильник|таймер|напоминание)\b", command):
        return _say(
            "Системный будильник, таймер или напоминание я из навыка не устанавливаю. "
            "Скажите «выход», затем обратитесь с просьбой к Алисе.",
            state,
        )

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return _say("В навыке ещё не настроен ключ Claude API.")

    if not utterance:
        return _say("Здравствуйте! Задайте вопрос, и я постараюсь ответить.", state)

    question = utterance[:1000]
    search = _needs_web_search(command)
    context = ""
    previous_question = state.get("last_question")
    previous_answer = state.get("last_answer")
    if isinstance(previous_question, str) and isinstance(previous_answer, str):
        context = (
            "Контекст предыдущего обмена (он может быть сокращён):\n"
            f"Пользователь: {previous_question}\nПомощник: {previous_answer}\n\n"
        )
    payload = {
        "model": MODEL,
        "system": INSTRUCTIONS + (" " + SEARCH_PROMPT if search else ""),
        "messages": [{"role": "user", "content": context + question}],
        "max_tokens": 220 if search else 120,
    }
    if search:
        payload["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}]
    try:
        data = _api_request(key, payload, timeout=CLAUDE_TIMEOUT_SECONDS)
        return _finish(data, question)
    except ClaudeError as exc:
        return _say(str(exc) + " Повторите вопрос позже.", state)


# HTTPS-вебхук Яндекс Диалогов для Vercel.
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlsplit
import hmac

SKILL_ID = os.environ.get("ALICE_SKILL_ID", "").strip()


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        expected = os.environ.get("ALICE_WEBHOOK_TOKEN", "")
        supplied = parse_qs(urlsplit(self.path).query).get("token", [""])[0]
        if not expected or not hmac.compare_digest(supplied, expected):
            return self._reply(404, {"error": "not found"})
        if not SKILL_ID:
            return self._reply(503, {"error": "ALICE_SKILL_ID is not configured"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 16384:
                return self._reply(413, {"error": "invalid request size"})
            event = json.loads(self.rfile.read(length))
            if not isinstance(event, dict) or (event.get("session") or {}).get("skill_id") != SKILL_ID:
                return self._reply(403, {"error": "invalid skill"})
            return self._reply(200, handler_alice(event, None))
        except (ValueError, json.JSONDecodeError):
            return self._reply(400, {"error": "invalid JSON"})

    def do_GET(self):
        self._reply(405, {"error": "method not allowed"})

    def _reply(self, status, body):
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass  # Не записывать секрет вебхука из URL в журнал приложения.

