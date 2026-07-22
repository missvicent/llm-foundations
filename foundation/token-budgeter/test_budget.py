from budget import count_tokens, chunks_that_fit


def test_count_is_real_tokens():
    assert count_tokens("hello world", "gpt-5.5") < len("hello world")


def test_small_window_drops_chunks():
    chunks = ["word " * 2000] * 30  # ~2k tokens each
    kept = chunks_that_fit(chunks, "qwen3.5-8b", "You are a helpful assistant.")
    assert 0 < len(kept) < 30
