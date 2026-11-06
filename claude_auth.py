"""claude code subscription auth für pyrate, direkt gegen api.anthropic.com.

liest oauth-credentials aus ~/.claude/.credentials.json (oder $CLAUDE_CONFIG_DIR),
auf macOS zusätzlich aus dem keychain. refresht den token bei bedarf direkt über
anthropic's oauth-endpoint.

übersetzt außerdem das protokoll: pyrate spricht openai chat/completions, anthropic
/v1/messages. dafür nutzt das plugin die erweiterte plugin-schnittstelle (pyrate >= 1.9.2):
`request()` biegt payload + url um und gibt die messages-url zurück, `response()`
bildet die anthropic-SSE-chunks zurück auf das openai-format.

aktivieren in ~/.config/pyrate/config.yaml:

    backends:
      - name: claude
        base_url: https://api.anthropic.com/v1
        plugin: ~/.config/pyrate/plugins/claude_auth.py
        models: [claude-sonnet-4-5, claude-opus-4-1]

oder mit einem proxy (z.b. CLIProxyAPI), dann ohne url-override nötig:

    backends:
      - name: claude
        base_url: http://localhost:8317/v1
        plugin: ~/.config/pyrate/plugins/claude_auth.py
"""

import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

# anthropic oauth (von claude code's login-flow, client_id ist öffentlich)
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"

# identity, die anthropic bei oauth-requests im system[] sehen will (sonst "not claude code")
SYSTEM_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."

EXPIRY_SKEW = 60   # s: so kurz vor ablauf wird schon refresht
CACHE_TTL = 30     # s: credentials-file/keychain nicht öfter lesen
DEFAULT_MAX_TOKENS = 32000
THINKING_BUDGET = 16000

_lock = threading.RLock()  # reentrant: _get_token hält das lock und ruft _load_credentials
_cache = {"creds": None, "at": 0.0}
# input_tokens: message_start-usage bis message_delta, pro request in request() reset.
# tool_names: tool_use_id → name aus content_block_start der response-chunks; wird NICHT
# pro request resettet, weil request() das mapping vom vorherigen response zum replay braucht.
_state = {"input_tokens": 0, "tool_names": {}}


class ClaudeAuthError(Exception):
    pass


def _config_dir():
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")


def _parse_credentials(raw):
    """→ dict mit accessToken/refreshToken/expiresAt oder None."""
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    creds = data.get("claudeAiOauth", data)
    if not isinstance(creds, dict):
        return None
    access = creds.get("accessToken")
    refresh = creds.get("refreshToken")
    expires = creds.get("expiresAt")
    if not (isinstance(access, str) and access and isinstance(refresh, str) and refresh):
        return None
    # expiresAt ist in ms
    exp = expires / 1000 if isinstance(expires, (int, float)) and expires > 1e12 else expires
    return {"access": access, "refresh": refresh, "expires": exp}


def _read_keychain():
    """macOS: credentials aus dem login-keychain. None auf linux/fehler."""
    if os.uname().sysname != "Darwin":
        return None
    try:
        out = subprocess.run(
            ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return _parse_credentials(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _read_file():
    """credentials aus ~/.claude/.credentials.json (oder $CLAUDE_CONFIG_DIR)."""
    path = _config_dir() / ".credentials.json"
    try:
        return _parse_credentials(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _load_credentials(force=False):
    """gültige credentials, cached. keychain wird file vorgezogen (aktueller)."""
    with _lock:
        now = time.time()
        if not force and _cache["creds"] and now - _cache["at"] < CACHE_TTL:
            return _cache["creds"]
        creds = _read_keychain() or _read_file()
        if creds:
            _cache["creds"] = creds
            _cache["at"] = now
        return creds


def _save_credentials(creds):
    """nach refresh: zurück in die datei schreiben (keychain macht das system)."""
    path = _config_dir() / ".credentials.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    blob = raw.get("claudeAiOauth", raw)
    if isinstance(blob, dict):
        blob["accessToken"] = creds["access"]
        blob["refreshToken"] = creds["refresh"]
        blob["expiresAt"] = int(creds["expires"] * 1000)
        raw["claudeAiOauth"] = blob
    else:
        raw = {"claudeAiOauth": {
            "accessToken": creds["access"],
            "refreshToken": creds["refresh"],
            "expiresAt": int(creds["expires"] * 1000),
        }}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        path.chmod(0o600)
    except OSError:
        pass  # nicht fatal: nächster refreshversuch klappt trotzdem


def _refresh(creds):
    """neuen access-token holen. wirft ClaudeAuthError bei fehler."""
    body = json.dumps({
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": creds["refresh"],
    }).encode()
    req = urllib.request.Request(TOKEN_URL, data=body, headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
    }, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise ClaudeAuthError(f"token-refresh fehlgeschlagen (http {e.code}): {detail}") from e
    except OSError as e:
        raise ClaudeAuthError(f"token-refresh: nicht erreichbar ({e})") from e
    access = data.get("access_token")
    refresh = data.get("refresh_token") or creds["refresh"]
    if not access:
        raise ClaudeAuthError(f"token-refresh: keine access_token in antwort: {data}")
    expires_in = data.get("expires_in")
    expires = time.time() + expires_in if isinstance(expires_in, (int, float)) else creds["expires"]
    return {"access": access, "refresh": refresh, "expires": expires}


def _get_token():
    """gültiger access-token, refresht bei bedarf. thread-safe."""
    creds = _load_credentials()
    if creds is None:
        raise ClaudeAuthError(
            f"keine claude-credentials gefunden ({_config_dir()}/.credentials.json). "
            "einmal `claude` starten und einloggen."
        )
    if creds["expires"] - EXPIRY_SKEW < time.time():
        with _lock:
            # doppel-check: ein anderer thread könnte schon refresht haben
            fresh = _load_credentials(force=True)
            if fresh and fresh["expires"] - EXPIRY_SKEW > time.time():
                creds = fresh
            else:
                creds = _refresh(creds)
                _cache["creds"] = creds
                _cache["at"] = time.time()
                _save_credentials(creds)
    return creds["access"]


# ---------------------------------------------------------------- chat/completions → /v1/messages

def _image_part(part):
    """openai image_url data-url → anthropic image block."""
    url = part.get("image_url", {}).get("url", "")
    head, _, data = url.partition(",")
    media = head.removeprefix("data:").split(";")[0] or "image/png"
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}


def _content_to_blocks(content):
    """openai content (str oder part-liste) → anthropic content-blocks."""
    if isinstance(content, str):
        return content
    out = []
    for p in content:
        if p.get("type") == "text":
            out.append({"type": "text", "text": p.get("text", "")})
        elif p.get("type") == "image_url":
            out.append(_image_part(p))
    return out


def _message_to_anthropic(m):
    """eine pyrate-message → anthropic-message(s). tool-messages werden vom aufrufer
    gesammelt (mehrere hintereinander → eine user-message mit mehreren tool_result)."""
    role, content = m.get("role"), m.get("content")
    if role == "user":
        return {"role": "user", "content": _content_to_blocks(content)}
    if role == "assistant":
        blocks = []
        # thinking nur mit signature replayen – ohne lehnt anthropic den request ab
        for tb in m.get("thinking_blocks") or []:
            if tb.get("signature"):
                blocks.append({"type": "thinking", "thinking": tb.get("thinking", ""),
                               "signature": tb["signature"]})
        if content:
            blocks.append({"type": "text", "text": content})
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            # name kann in pyrate's history leer sein (aus deltas rekonstruiert) →
            # aus dem beim streaming gemerkten id→name-mapping holen
            name = fn.get("name") or _state["tool_names"].get(c.get("id")) or ""
            blocks.append({"type": "tool_use", "id": c.get("id") or "",
                           "name": name, "input": args})
        return {"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]}
    return None  # system/tool behandeln request() direkt


def _tools_to_anthropic(tools):
    return [{"name": t["function"]["name"],
             "description": t["function"].get("description") or "",
             "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}}}
            for t in tools or []]


def _transform_payload(payload):
    """openai chat/completions payload → anthropic messages payload (in-place-ish:
    messages-liste wird neu gebaut, restliche keys gesetzt/gelöscht)."""
    _state["input_tokens"] = 0
    out_msgs = []
    system_texts = []
    pending_results = []  # tool_result-blocks, sammeln bis zur nächsten non-tool-message

    def flush_results():
        if pending_results:
            out_msgs.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for m in payload.get("messages") or []:
        role = m.get("role")
        if role == "system":
            system_texts.append(m.get("content") if isinstance(m.get("content"), str)
                                else " ".join(p.get("text", "") for p in m.get("content") or []))
            continue
        if role == "tool":
            pending_results.append({"type": "tool_result", "tool_use_id": m.get("tool_call_id"),
                                    "content": m.get("content") or "", "is_error": False})
            continue
        flush_results()
        am = _message_to_anthropic(m)
        if am:
            out_msgs.append(am)
    flush_results()

    payload["messages"] = out_msgs
    # anthropic will die identity als eigenen system-eintrag, fremde system-prompts
    # werden bei oauth-abrechnung abgelehnt → rest landet in der ersten user-message
    payload["system"] = [{"type": "text", "text": SYSTEM_IDENTITY}]
    if system_texts:
        prefix = "\n\n".join(t for t in system_texts if t)
        if prefix:
            first_user = next((m for m in out_msgs if m["role"] == "user"), None)
            if first_user is not None:
                c = first_user["content"]
                if isinstance(c, str):
                    first_user["content"] = prefix + "\n\n" + c
                else:
                    first_user["content"] = [{"type": "text", "text": prefix}, *c]
            else:
                out_msgs.insert(0, {"role": "user", "content": prefix})
    if payload.get("tools"):
        payload["tools"] = _tools_to_anthropic(payload["tools"])
    # letzter fallback: tool_use mit leerem name bekäme von anthropic ein http 400
    # ("String should have at least 1 character") → deklarierten tool-namen einsetzen
    fallback_name = next((t["name"] for t in payload.get("tools") or []), "tool_use")
    for am in out_msgs:
        if am["role"] != "assistant" or not isinstance(am["content"], list):
            continue
        for b in am["content"]:
            if b.get("type") == "tool_use" and not b.get("name"):
                b["name"] = _state["tool_names"].get(b.get("id")) or fallback_name
    payload.pop("stream_options", None)  # openai-only
    payload.setdefault("max_tokens", DEFAULT_MAX_TOKENS)  # pflichtfeld bei anthropic
    effort = payload.pop("reasoning_effort", None)
    if effort and effort != "low":
        payload["thinking"] = {"type": "enabled",
                               "budget_tokens": min(THINKING_BUDGET, payload["max_tokens"] // 2)}
    return payload


# ---------------------------------------------------------------- /v1/messages → chat/completions chunks

_STOP_REASONS = {"end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop",
                 "tool_use": "tool_calls", "refusal": "stop"}


def response(chunk):
    """pyrate plugin-hook: anthropic-SSE-event → openai-chunk oder None."""
    t = chunk.get("type")
    if t == "message_start":
        usage = (chunk.get("message") or {}).get("usage") or {}
        _state["input_tokens"] = usage.get("input_tokens", 0)
        return None
    if t == "content_block_start":
        # anthropic hängt den block am key "content_block" (nicht "block")
        block = chunk.get("content_block") or chunk.get("block") or {}
        if block.get("type") == "tool_use":
            bid, bname = block.get("id") or "", block.get("name") or ""
            if bid and bname:
                _state["tool_names"][bid] = bname  # für den replay im nächsten request
            return {"choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": chunk.get("index", 0), "id": bid, "type": "function",
                "function": {"name": bname, "arguments": ""}}]}}]}
        return None
    if t == "content_block_delta":
        d = chunk.get("delta") or {}
        dt = d.get("type")
        if dt == "text_delta":
            return {"choices": [{"index": 0, "delta": {"content": d.get("text", "")}}]}
        if dt == "thinking_delta":
            return {"choices": [{"index": 0, "delta": {"reasoning_content": d.get("thinking", "")}}]}
        if dt == "signature_delta":
            return {"choices": [{"index": 0, "delta": {"thinking_blocks": [
                {"type": "thinking", "signature": d.get("signature", "")}]}}]}
        if dt == "input_json_delta":
            return {"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": chunk.get("index", 0), "function": {"arguments": d.get("partial_json", "")}}]}}]}
        return None
    if t == "message_delta":
        d = chunk.get("delta") or {}
        usage = chunk.get("usage") or {}
        out = usage.get("output_tokens", 0)
        chunk_out = {"choices": [{"index": 0, "delta": {},
                                  "finish_reason": _STOP_REASONS.get(d.get("stop_reason"), "stop")}],
                     "usage": {"prompt_tokens": _state.get("input_tokens", 0),
                               "completion_tokens": out,
                               "total_tokens": _state.get("input_tokens", 0) + out}}
        return chunk_out
    if t == "error":
        return chunk  # pyrate wirft selbst LLMError bei chunk["error"]
    return None  # ping, content_block_stop, message_stop, ...


def request(backend, headers, payload, session):
    """pyrate plugin-hook: oauth-token setzen, payload auf /v1/messages biegen,
    messages-url zurückgeben."""
    token = _get_token()
    headers["Authorization"] = f"Bearer {token}"
    headers.pop("x-api-key", None)
    headers.setdefault("anthropic-version", "2023-06-01")
    headers["anthropic-dangerous-direct-browser-access"] = "true"
    headers["x-app"] = "cli"
    headers.setdefault("User-Agent", "claude-cli/2.0.0 (external, sdk-cli)")
    # betas für claude code features (oauth, prompt-caching, interleaved thinking)
    beta = headers.get("anthropic-beta", "")
    needed = {"claude-code-20250219", "oauth-2025-04-20", "interleaved-thinking-2025-05-14"}
    have = {b.strip() for b in beta.split(",") if b.strip()}
    headers["anthropic-beta"] = ",".join(sorted(needed | have))
    _transform_payload(payload)
    return f"{backend.base_url}/messages?beta=true"
