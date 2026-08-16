"""Persistent local diagnostics for crash and performance investigation."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import threading
import time
from typing import Mapping

from platformdirs import user_log_path

_LOG_NAME = "ditherzam.log"
_LOGGER = logging.getLogger(__name__)
_started_at = time.monotonic()
_configured_path: Path | None = None
_ACTION_LOGGER = logging.getLogger("ditherzam.action")

_MAX_FIELDS = 32
_MAX_KEY_CHARS = 64
_MAX_STRING_CHARS = 256
_SENSITIVE_FIELD_PARTS = frozenset({
    "array", "bitmap", "bytes", "content", "data", "image", "mask_data",
    "payload", "pixels", "preset", "tensor",
})
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class SemanticActionLog:
    """Thread-safe, low-overhead semantic action recorder.

    Gesture updates are coalesced in memory and emit one record on ``end()``.
    Action IDs are assigned while emitting, so IDs always increase in log order.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or _ACTION_LOGGER
        self._lock = threading.Lock()
        self._next_action_id = 1
        self._next_gesture_token = 1
        self._gestures: dict[int, tuple[str, dict[str, object]]] = {}

    def record(self, action: str, **fields: object) -> int:
        """Immediately emit one completed semantic action."""
        with self._lock:
            return self._emit_locked(action, fields)

    def begin(self, action: str, **fields: object) -> int:
        """Begin a coalesced gesture and return its process-local token."""
        with self._lock:
            token = self._next_gesture_token
            self._next_gesture_token += 1
            self._gestures[token] = (str(action), dict(fields))
            return token

    def update(self, token: int, **fields: object) -> bool:
        """Replace a gesture's latest values without writing a log record."""
        with self._lock:
            gesture = self._gestures.get(token)
            if gesture is None:
                return False
            gesture[1].update(fields)
            return True

    def end(self, token: int, **fields: object) -> int | None:
        """Finish a gesture and emit its single coalesced action record."""
        with self._lock:
            gesture = self._gestures.pop(token, None)
            if gesture is None:
                return None
            action, accumulated = gesture
            accumulated.update(fields)
            return self._emit_locked(action, accumulated)

    def cancel(self, token: int) -> bool:
        """Discard a gesture without emitting it."""
        with self._lock:
            return self._gestures.pop(token, None) is not None

    def _emit_locked(self, action: str, fields: Mapping[str, object]) -> int:
        action_id = self._next_action_id
        self._next_action_id += 1
        record = {
            "action": _bounded_text(action),
            "action_id": action_id,
            "fields": _safe_fields(fields),
        }
        payload = json.dumps(
            record, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        self._logger.info("ACTION %s", payload)
        return action_id


semantic_actions = SemanticActionLog()


def log_action(action: str, **fields: object) -> int:
    """Record one immediate semantic action with the shared recorder."""
    return semantic_actions.record(action, **fields)


def begin_action(action: str, **fields: object) -> int:
    """Begin a coalesced action with the shared recorder."""
    return semantic_actions.begin(action, **fields)


def update_action(token: int, **fields: object) -> bool:
    """Update a shared coalesced action without emitting."""
    return semantic_actions.update(token, **fields)


def end_action(token: int, **fields: object) -> int | None:
    """Emit a shared coalesced action."""
    return semantic_actions.end(token, **fields)


def cancel_action(token: int) -> bool:
    """Discard a shared coalesced action."""
    return semantic_actions.cancel(token)


def _safe_fields(fields: Mapping[str, object]) -> dict[str, object]:
    safe: dict[str, object] = {}
    for index, (raw_key, value) in enumerate(
        sorted(fields.items(), key=lambda item: str(item[0]))
    ):
        if index >= _MAX_FIELDS:
            safe["_omitted_fields"] = len(fields) - _MAX_FIELDS
            break
        key = _bounded_text(raw_key, _MAX_KEY_CHARS)
        safe[key] = _safe_value(key, value)
    return safe


def _safe_value(key: str, value: object) -> object:
    lowered = key.casefold()
    if any(part in lowered for part in _SENSITIVE_FIELD_PARTS):
        return _redacted_summary(value)
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return _path_summary(value)
    if isinstance(value, str):
        if _looks_absolute_path(value):
            return _path_summary(Path(value))
        return _bounded_text(value)
    if isinstance(value, Mapping):
        return _safe_fields(value)
    if isinstance(value, (list, tuple, set, frozenset, bytes, bytearray, memoryview)):
        return _redacted_summary(value)
    if hasattr(value, "shape") or hasattr(value, "dtype"):
        return _redacted_summary(value)
    return {"redacted": type(value).__name__}


def _redacted_summary(value: object) -> dict[str, object]:
    summary: dict[str, object] = {"redacted": type(value).__name__}
    try:
        summary["length"] = len(value)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        pass
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            summary["shape"] = "x".join(str(int(part)) for part in shape)
        except (TypeError, ValueError):
            pass
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        summary["dtype"] = _bounded_text(dtype, 32)
    return summary


def _path_summary(path: Path) -> dict[str, object]:
    return {
        "basename": _bounded_text(path.name),
        "extension": _bounded_text(path.suffix, 32),
        "redacted": "path",
    }


def _looks_absolute_path(value: str) -> bool:
    return value.startswith(("/", "\\\\")) or bool(_WINDOWS_ABSOLUTE_PATH.match(value))


def _bounded_text(value: object, limit: int = _MAX_STRING_CHARS) -> str:
    text = str(value).replace("\r", "\\r").replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "..."


def default_log_path() -> Path:
    return Path(user_log_path("ditherzam", appauthor=False)) / _LOG_NAME


def configure_logging(log_dir: str | Path | None = None) -> Path:
    """Install one bounded UTF-8 application log and return its path."""
    global _configured_path
    folder = Path(log_dir) if log_dir is not None else default_log_path().parent
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / _LOG_NAME
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_ditherzam_diagnostics", False):
            root.removeHandler(handler)
            handler.close()
    handler = RotatingFileHandler(
        path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler._ditherzam_diagnostics = True  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s %(threadName)s "
        "%(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    _configured_path = path
    _LOGGER.info(
        "session_start version=%s python=%s platform=%s pid=%d",
        _app_version(), platform.python_version(), platform.platform(), os.getpid(),
    )
    return path


def install_exception_hooks() -> None:
    """Record uncaught exceptions while preserving normal error reporting."""
    previous_sys = sys.excepthook
    previous_thread = threading.excepthook

    def sys_hook(exc_type, exc_value, traceback) -> None:
        logging.getLogger("ditherzam.crash").critical(
            "uncaught_main_exception", exc_info=(exc_type, exc_value, traceback)
        )
        previous_sys(exc_type, exc_value, traceback)

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        logging.getLogger("ditherzam.crash").critical(
            "uncaught_thread_exception thread=%s",
            getattr(args.thread, "name", "unknown"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        previous_thread(args)

    sys.excepthook = sys_hook
    threading.excepthook = thread_hook


def log_duration(
    logger: logging.Logger, operation: str, started: float, **fields: object
) -> None:
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info(
        "%s duration_ms=%.1f%s",
        operation, (time.perf_counter() - started) * 1000.0,
        f" {details}" if details else "",
    )


def read_log_tail(path: str | Path, tail_chars: int = 80_000) -> str:
    log_path = Path(path)
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - tail_chars))
            text = stream.read()
        return text if size <= tail_chars else "[earlier log omitted]\n" + text
    except FileNotFoundError:
        return "(No program log has been written yet.)"
    except OSError as exc:
        return f"(Could not read program log: {exc})"


def build_diagnostic_report(
    path: str | Path | None = None, *, tail_chars: int = 80_000
) -> str:
    log_path = Path(path) if path is not None else (_configured_path or default_log_path())
    return "\n".join([
        "ditherzam diagnostic report",
        f"Version: {_app_version()}",
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        f"Process: {os.getpid()}",
        f"Session uptime: {time.monotonic() - _started_at:.1f}s",
        f"Log file: {log_path}",
        "",
        "Recent program notes:",
        read_log_tail(log_path, tail_chars),
    ])


def clear_logs(path: str | Path) -> None:
    log_path = Path(path)
    try:
        log_path.write_text("", encoding="utf-8")
    except FileNotFoundError:
        pass
    for candidate in (Path(f"{log_path}.{i}") for i in range(1, 4)):
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass


def _app_version() -> str:
    try:
        from ditherzam import __version__
        return __version__
    except Exception:
        return "development"
