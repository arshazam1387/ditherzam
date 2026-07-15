from concurrent.futures import ThreadPoolExecutor

import pytest

from ditherzam.masking.session import LazySession


def test_lazy_session_creates_once_and_reuses_across_threads():
    calls = []
    value = object()

    def factory():
        calls.append(1)
        return value

    holder = LazySession(factory)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: holder.get(), range(40)))
    assert all(result is value for result in results)
    assert calls == [1]


def test_failed_creation_is_not_cached():
    calls = []

    def factory():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("missing")
        return "ready"

    holder = LazySession(factory)
    with pytest.raises(RuntimeError, match="missing"):
        holder.get()
    assert holder.get() == "ready"
    assert len(calls) == 2


def test_none_session_fails_honestly():
    with pytest.raises(RuntimeError, match="returned None"):
        LazySession(lambda: None).get()
