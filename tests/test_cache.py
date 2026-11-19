from ideaforge.llm.cache_helper import MessageCache
from ideaforge.llm.clients import ChatMessage


def test_same_prefix_hits_cache():
    cache = MessageCache()
    msgs = [ChatMessage(role="user", content="hello")]
    key = MessageCache.key("fixer", "sys", msgs)
    cache.put(key, "cached-value")
    assert cache.get(key) == "cached-value"
    assert cache.hits == 1
    assert cache.misses == 0


def test_different_prefix_misses():
    cache = MessageCache()
    key1 = MessageCache.key("fixer", "sys", [ChatMessage(role="user", content="a")])
    key2 = MessageCache.key("fixer", "sys", [ChatMessage(role="user", content="b")])
    cache.put(key1, "v1")
    assert cache.get(key2) is None
    assert cache.misses == 1
    key3 = MessageCache.key("fixer", "other-system", [ChatMessage(role="user", content="a")])
    assert cache.get(key3) is None


def test_role_and_temperature_in_key():

    msgs = [ChatMessage(role="user", content="x")]
    k1 = MessageCache.key("fixer", None, msgs)
    k2 = MessageCache.key("judge_final", None, msgs)
    assert k1 != k2
    k3 = MessageCache.key("fixer", None, msgs, temperature=0.9)
    assert k1 != k3