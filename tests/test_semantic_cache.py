from src.services.semantic_cache import SemanticCache


def test_semantic_cache_reuses_similar_questions(tmp_path):
    cache = SemanticCache(str(tmp_path / "cache.sqlite3"), threshold=0.7)
    cache.put("How many years of experience do you have?", "resume-v1", "Six years.")

    cached = cache.lookup("How many years experience do you have", "resume-v1")
    assert cached is not None
    assert cached["response"] == "Six years."
    assert cache.lookup("What is your favorite color?", "resume-v1") is None


def test_semantic_cache_stores_audio(tmp_path):
    cache = SemanticCache(str(tmp_path / "cache.sqlite3"), threshold=0.7)
    cache.put("Tell me about Agent Builder", "resume-v1", "It reduced latency.")
    cache.update_audio("Tell me about Agent Builder", "resume-v1", b"wav", "audio/wav")

    cached = cache.lookup("Tell me about Agent Builder", "resume-v1")
    assert cached["audio"] == b"wav"
    assert cached["audio_format"] == "audio/wav"