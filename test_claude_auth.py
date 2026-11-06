"""claude_auth plugin: credentials laden, token setzen, refresh bei ablauf."""

import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

PLUGIN_PATH = Path(__file__).parent.parent / "plugins" / "claude_auth.py"


def load_plugin():
    spec = importlib.util.spec_from_file_location("claude_auth_test", PLUGIN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def claude_dir(tmp_path, monkeypatch):
    """isoliertes $CLAUDE_CONFIG_DIR mit gefakten credentials."""
    d = tmp_path / "claude"
    d.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(d))
    return d


def write_creds(claude_dir, *, access="sk-ant-oat01-test", refresh="rt-test",
                expires_in=3600, subscription="team"):
    expires_at = int((time.time() + expires_in) * 1000)
    (claude_dir / ".credentials.json").write_text(json.dumps({
        "claudeAiOauth": {
            "accessToken": access,
            "refreshToken": refresh,
            "expiresAt": expires_at,
            "subscriptionType": subscription,
        }
    }), encoding="utf-8")


def make_backend():
    return SimpleNamespace(id="claude", base_url="https://api.anthropic.com/v1",
                           api_key="sk-fake")


def test_request_sets_bearer_and_strips_api_key(claude_dir):
    write_creds(claude_dir, access="sk-ant-oat01-abc123")
    mod = load_plugin()
    headers = {"Authorization": "Bearer sk-fake", "x-api-key": "sk-fake"}
    mod.request(make_backend(), headers, {}, None)
    assert headers["Authorization"] == "Bearer sk-ant-oat01-abc123"
    assert "x-api-key" not in headers


def test_request_sets_anthropic_headers(claude_dir):
    write_creds(claude_dir)
    mod = load_plugin()
    headers = {}
    mod.request(make_backend(), headers, {}, None)
    assert headers["anthropic-version"] == "2023-06-01"
    beta = headers["anthropic-beta"]
    assert "oauth-2025-04-20" in beta
    assert "claude-code-20250219" in beta


def test_request_preserves_existing_betas(claude_dir):
    write_creds(claude_dir)
    mod = load_plugin()
    headers = {"anthropic-beta": "custom-beta-123"}
    mod.request(make_backend(), headers, {}, None)
    beta = headers["anthropic-beta"]
    assert "custom-beta-123" in beta
    assert "oauth-2025-04-20" in beta


def test_no_credentials_raises(claude_dir):
    mod = load_plugin()
    with pytest.raises(Exception, match="keine claude-credentials"):
        mod.request(make_backend(), {}, {}, None)


def test_expired_token_triggers_refresh(claude_dir, monkeypatch):
    """abgelaufener token → refresh über oauth-endpoint, neuer token landet im header."""
    write_creds(claude_dir, access="sk-ant-old", refresh="rt-old", expires_in=-10)
    mod = load_plugin()

    refreshed = {"access_token": "sk-ant-new", "refresh_token": "rt-new", "expires_in": 3600}

    class FakeResp:
        def __init__(self, data):
            self._data = data
        def read(self):
            return json.dumps(self._data).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    calls = []
    def fake_urlopen(req, timeout=None):
        calls.append(req)
        return FakeResp(refreshed)

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)

    headers = {}
    mod.request(make_backend(), headers, {}, None)
    assert headers["Authorization"] == "Bearer sk-ant-new"
    assert len(calls) == 1
    body = json.loads(calls[0].data)
    assert body["grant_type"] == "refresh_token"
    assert body["refresh_token"] == "rt-old"


def test_refresh_writes_back_to_file(claude_dir, monkeypatch):
    """nach refresh landet der neue token in .credentials.json."""
    write_creds(claude_dir, access="sk-ant-old", refresh="rt-old", expires_in=-10)
    mod = load_plugin()

    class FakeResp:
        def read(self):
            return json.dumps({"access_token": "sk-ant-new", "refresh_token": "rt-new",
                               "expires_in": 3600}).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda req, timeout=None: FakeResp())
    mod.request(make_backend(), {}, {}, None)

    saved = json.loads((claude_dir / ".credentials.json").read_text())
    assert saved["claudeAiOauth"]["accessToken"] == "sk-ant-new"
    assert saved["claudeAiOauth"]["refreshToken"] == "rt-new"


def test_valid_token_no_refresh(claude_dir, monkeypatch):
    """gültiger token → kein refresh-call."""
    write_creds(claude_dir, access="sk-ant-valid", expires_in=3600)
    mod = load_plugin()

    def boom(*a, **kw):
        raise AssertionError("kein refresh bei gültigem token")

    monkeypatch.setattr(mod.urllib.request, "urlopen", boom)
    headers = {}
    mod.request(make_backend(), headers, {}, None)
    assert headers["Authorization"] == "Bearer sk-ant-valid"


def test_credentials_cached(claude_dir, monkeypatch):
    """zweimal request kurz hintereinander → nur ein file-read."""
    write_creds(claude_dir, access="sk-ant-cached")
    mod = load_plugin()

    reads = []
    orig = mod._read_file
    def counting():
        reads.append(1)
        return orig()
    monkeypatch.setattr(mod, "_read_file", counting)

    mod.request(make_backend(), {}, {}, None)
    mod.request(make_backend(), {}, {}, None)
    assert len(reads) == 1


def test_parse_credentials_flat_format(claude_dir):
    """auch ohne claudeAiOauth-wrapper (flat) parsen."""
    (claude_dir / ".credentials.json").write_text(json.dumps({
        "accessToken": "sk-ant-flat",
        "refreshToken": "rt-flat",
        "expiresAt": int((time.time() + 3600) * 1000),
    }), encoding="utf-8")
    mod = load_plugin()
    headers = {}
    mod.request(make_backend(), headers, {}, None)
    assert headers["Authorization"] == "Bearer sk-ant-flat"


def test_parse_credentials_rejects_mcp_only(claude_dir):
    """mcpOAuth ohne accessToken → kein gültiges credential."""
    (claude_dir / ".credentials.json").write_text(json.dumps({
        "mcpOAuth": {"accessToken": "x"},
    }), encoding="utf-8")
    mod = load_plugin()
    with pytest.raises(Exception, match="keine claude-credentials"):
        mod.request(make_backend(), {}, {}, None)


# ---------------------------------------------------------------- protokoll-übersetzung

def call_request(mod, claude_dir, payload):
    write_creds(claude_dir)
    headers = {}
    url = mod.request(make_backend(), headers, payload, None)
    return headers, url, payload


def test_request_returns_messages_url(claude_dir):
    mod = load_plugin()
    _, url, _ = call_request(mod, claude_dir, {})
    assert url == "https://api.anthropic.com/v1/messages?beta=true"


def test_payload_system_goes_to_first_user(claude_dir):
    """system[] bekommt nur die identity, der eigene prompt wandert in die erste user-message."""
    mod = load_plugin()
    payload = {"messages": [
        {"role": "system", "content": "du bist pyrate."},
        {"role": "user", "content": "hi"},
    ]}
    _, _, out = call_request(mod, claude_dir, payload)
    assert out["system"] == [{"type": "text", "text": mod.SYSTEM_IDENTITY}]
    assert out["messages"] == [{"role": "user", "content": "du bist pyrate.\n\nhi"}]


def test_payload_assistant_tool_calls_become_tool_use(claude_dir):
    mod = load_plugin()
    payload = {"messages": [
        {"role": "user", "content": "mach"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "read", "arguments": '{"path": "/x"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "inhalt"},
        {"role": "tool", "tool_call_id": "call_2", "content": "anderes"},
    ]}
    _, _, out = call_request(mod, claude_dir, payload)
    assert out["messages"][1] == {"role": "assistant", "content": [
        {"type": "tool_use", "id": "call_1", "name": "read", "input": {"path": "/x"}}]}
    # zwei tool-messages → eine user-message mit zwei tool_result-blocks
    assert out["messages"][2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "inhalt", "is_error": False},
        {"type": "tool_result", "tool_use_id": "call_2", "content": "anderes", "is_error": False}]}


def test_payload_thinking_kept_only_with_signature(claude_dir):
    mod = load_plugin()
    payload = {"messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "fertig", "reasoning_content": "gedacht",
         "thinking_blocks": [{"type": "thinking", "thinking": "gedacht", "signature": "sig-1"}]},
        {"role": "assistant", "content": "ohne sig", "reasoning_content": "wegwerf"},
    ]}
    _, _, out = call_request(mod, claude_dir, payload)
    assert out["messages"][1]["content"] == [
        {"type": "thinking", "thinking": "gedacht", "signature": "sig-1"},
        {"type": "text", "text": "fertig"}]
    assert out["messages"][2]["content"] == [{"type": "text", "text": "ohne sig"}]


def test_payload_tools_and_effort(claude_dir):
    mod = load_plugin()
    payload = {"messages": [{"role": "user", "content": "hi"}], "stream": True,
               "stream_options": {"include_usage": True}, "reasoning_effort": "high",
               "tools": [{"type": "function", "function": {
                   "name": "read", "description": "liest", "parameters": {"type": "object"}}}]}
    _, _, out = call_request(mod, claude_dir, payload)
    assert out["tools"] == [{"name": "read", "description": "liest",
                             "input_schema": {"type": "object"}}]
    assert "stream_options" not in out
    assert out["max_tokens"] == mod.DEFAULT_MAX_TOKENS
    assert out["thinking"] == {"type": "enabled", "budget_tokens": mod.THINKING_BUDGET}


def test_payload_effort_low_disables_thinking(claude_dir):
    mod = load_plugin()
    payload = {"messages": [], "reasoning_effort": "low"}
    _, _, out = call_request(mod, claude_dir, payload)
    assert "thinking" not in out
    assert "reasoning_effort" not in out


def test_payload_image_data_url(claude_dir):
    mod = load_plugin()
    payload = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "schau"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}]}]}
    _, _, out = call_request(mod, claude_dir, payload)
    assert out["messages"][0]["content"] == [
        {"type": "text", "text": "schau"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}}]


def test_response_text_and_finish(claude_dir):
    mod = load_plugin()
    call_request(mod, claude_dir, {})  # reset _state
    assert mod.response({"type": "message_start", "message": {"usage": {"input_tokens": 1234}}}) is None
    assert mod.response({"type": "content_block_delta", "index": 0,
                         "delta": {"type": "text_delta", "text": "hallo"}}) == \
        {"choices": [{"index": 0, "delta": {"content": "hallo"}}]}
    assert mod.response({"type": "ping"}) is None
    fin = mod.response({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 42}})
    assert fin["choices"][0]["finish_reason"] == "stop"
    assert fin["usage"] == {"prompt_tokens": 1234, "completion_tokens": 42, "total_tokens": 1276}


def test_response_tool_use_stream(claude_dir):
    mod = load_plugin()
    call_request(mod, claude_dir, {})
    start = mod.response({"type": "content_block_start", "index": 0,
                          "block": {"type": "tool_use", "id": "toolu_1", "name": "read"}})
    assert start["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "read"
    assert start["choices"][0]["delta"]["tool_calls"][0]["id"] == "toolu_1"
    part = mod.response({"type": "content_block_delta", "index": 0,
                         "delta": {"type": "input_json_delta", "partial_json": '{"path":'}})
    assert part["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"path":'
    fin = mod.response({"type": "message_delta", "delta": {"stop_reason": "tool_use"},
                        "usage": {"output_tokens": 5}})
    assert fin["choices"][0]["finish_reason"] == "tool_calls"


def test_response_content_block_start_uses_content_block_key(claude_dir):
    """anthropic schickt den block am key 'content_block' (nicht 'block') – sonst
    landet name/id nie im tool_call und pyrate meldet `unknown tool ''`."""
    mod = load_plugin()
    call_request(mod, claude_dir, {})
    start = mod.response({"type": "content_block_start", "index": 1,
                          "content_block": {"type": "tool_use", "id": "toolu_7", "name": "bash"}})
    tc = start["choices"][0]["delta"]["tool_calls"][0]
    assert tc["id"] == "toolu_7"
    assert tc["function"]["name"] == "bash"
    assert mod._state["tool_names"]["toolu_7"] == "bash"


def test_response_thinking_and_error(claude_dir):
    mod = load_plugin()
    call_request(mod, claude_dir, {})
    think = mod.response({"type": "content_block_delta", "index": 0,
                          "delta": {"type": "thinking_delta", "thinking": "hmm"}})
    assert think["choices"][0]["delta"]["reasoning_content"] == "hmm"
    sig = mod.response({"type": "content_block_delta", "index": 0,
                        "delta": {"type": "signature_delta", "signature": "sig"}})
    assert sig["choices"][0]["delta"]["thinking_blocks"][0]["signature"] == "sig"
    err = {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}
    assert mod.response(err) is err  # pyrate wirft selbst


def test_response_tool_use_registers_name(claude_dir):
    """content_block_start merkt sich id→name für den replay im nächsten request."""
    mod = load_plugin()
    call_request(mod, claude_dir, {})
    mod.response({"type": "content_block_start", "index": 0,
                  "block": {"type": "tool_use", "id": "toolu_1", "name": "read"}})
    assert mod._state["tool_names"]["toolu_1"] == "read"


def test_payload_tool_use_empty_name_uses_streamed_name(claude_dir):
    """pyrate hat den namen aus den deltas verloren → mapping aus content_block_start."""
    mod = load_plugin()
    call_request(mod, claude_dir, {})
    mod.response({"type": "content_block_start", "index": 0,
                  "block": {"type": "tool_use", "id": "toolu_9", "name": "bash"}})
    payload = {"messages": [
        {"role": "user", "content": "mach"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "toolu_9", "type": "function",
             "function": {"name": "", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "toolu_9", "content": "ok"},
    ]}
    _, _, out = call_request(mod, claude_dir, payload)
    assert out["messages"][1]["content"][0] == {
        "type": "tool_use", "id": "toolu_9", "name": "bash", "input": {}}


def test_payload_tool_use_empty_name_falls_back_to_declared_tool(claude_dir):
    """kein mapping da → erster deklarierter tool-name, nie leer (sonst http 400)."""
    mod = load_plugin()
    payload = {"messages": [
        {"role": "user", "content": "mach"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_x", "type": "function",
             "function": {"name": "", "arguments": ""}}]},
    ], "tools": [{"type": "function", "function": {"name": "read", "parameters": {}}}]}
    _, _, out = call_request(mod, claude_dir, payload)
    block = out["messages"][1]["content"][0]
    assert block["type"] == "tool_use"
    assert block["name"] == "read"


def test_payload_tool_use_missing_function_no_crash(claude_dir):
    """function=None/fehlend darf nicht mit KeyError platzen."""
    mod = load_plugin()
    payload = {"messages": [
        {"role": "user", "content": "mach"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_y", "type": "function", "function": None}]},
    ]}
    _, _, out = call_request(mod, claude_dir, payload)
    block = out["messages"][1]["content"][0]
    assert block["type"] == "tool_use"
    assert block["input"] == {}
    assert block["name"]  # nie leer
