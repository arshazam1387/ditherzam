"""Synchronized lazy holder for one validated local inference session."""
from __future__ import annotations

from threading import Lock
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class LazySession(Generic[T]):
    def __init__(self, factory: Callable[[], T]) -> None:
        if not callable(factory):
            raise TypeError("session factory must be callable")
        self._factory = factory
        self._session: T | None = None
        self._lock = Lock()

    def get(self) -> T:
        session = self._session
        if session is not None:
            return session
        with self._lock:
            if self._session is None:
                created = self._factory()
                if created is None:
                    raise RuntimeError("session factory returned None")
                self._session = created
            return self._session
