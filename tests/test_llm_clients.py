import httpx
import pytest

from ideaforge.errors import LLMAuthError, LLMTimeoutError, LLMTransportError
from ideaforge.llm.clients import ChatMessage, OpenAICompatClient
from ideaforge.llm.reasoning import is_reasoning_model, strip_reasoning_content


def api_response(content: str, model="gpt-4o", prompt=11, completion=7) -> dict:
    return {
        "model": model,
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


def patch_http(monkeypatch, handler):
    def fake_post(url, **kwargs):
        return handler(url, **kwargs)

    monkeypatch.setattr("ideaforge.llm.clients.httpx.post", fake_post)


def make_client(**kwargs) -> OpenAICompatClient:
    return OpenAICompatClient(
        model="gpt-4o", base_url="https://api.openai.com/v1",
        api_key="sk-test", api_key_env="OPENAI_API_KEY", **kwargs
    )


def test_usage_returned(monkeypatch):
    patch_http(monkeypatch, lambda url, **kw: httpx.Response(
        200, json=api_response("hello"), request=httpx.Request("POST", url)))
    resp = make_client().complete([ChatMessage(role="user", content="hi")])
    assert resp.content == "hello"
    assert resp.usage.prompt_tokens == 11
    assert resp.usage.completion_tokens == 7
    assert resp.model == "gpt-4o"


def test_reasoning_content_stripped(monkeypatch):
    raw = "<think>internal scratchpad</think>The answer is 42."
    patch_http(monkeypatch, lambda url, **kw: httpx.Response(
        200, json=api_response(raw), request=httpx.Request("POST", url)))
    resp = make_client().complete([ChatMessage(role="user", content="hi")])
    assert "scratchpad" not in resp.content
    assert "42" in resp.content


def test_auth_failure_no_retry(monkeypatch):
    patch_http(monkeypatch, lambda url, **kw: httpx.Response(
        401, json={}, request=httpx.Request("POST", url)))
    with pytest.raises(LLMAuthError):
        make_client().complete([ChatMessage(role="user", content="hi")])


def test_timeout_raises_llm_timeout(monkeypatch):
    def handler(url, **kw):
        raise httpx.ConnectTimeout("timed out")
    patch_http(monkeypatch, handler)
    with pytest.raises(LLMTimeoutError):
        make_client().complete([ChatMessage(role="user", content="hi")])


def test_transport_error_on_429(monkeypatch):
    patch_http(monkeypatch, lambda url, **kw: httpx.Response(
        429, json={}, request=httpx.Request("POST", url)))
    with pytest.raises(LLMTransportError):
        make_client().complete([ChatMessage(role="user", content="hi")])


def test_transport_error_on_500(monkeypatch):
    patch_http(monkeypatch, lambda url, **kw: httpx.Response(
        500, json={}, request=httpx.Request("POST", url)))
    with pytest.raises(LLMTransportError):
        make_client().complete([ChatMessage(role="user", content="hi")])


def test_missing_api_key_raises_auth_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAICompatClient(model="gpt-4o", api_key_env="OPENAI_API_KEY")
    with pytest.raises(LLMAuthError):
        client.complete([ChatMessage(role="user", content="hi")])


def test_strip_reasoning_edge_cases():
    assert strip_reasoning_content("<think>a</think>ok") == ("ok", "<think>a</think>")
    assert strip_reasoning_content("plain") == ("plain", "")
    _cleaned, reasoning = strip_reasoning_content("<think>unterminated ...")
    assert reasoning != ""
    cleaned2, _ = strip_reasoning_content("</think>tail")
    assert cleaned2 == "tail"
    assert is_reasoning_model("deepseek-reasoner")
    assert not is_reasoning_model("deepseek-chat")