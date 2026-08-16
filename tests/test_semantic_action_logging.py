import json
import logging
import threading

from ditherzam.diagnostics import SemanticActionLog


class _MessageHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _recorder():
    logger = logging.Logger("semantic-action-test")
    handler = _MessageHandler()
    logger.addHandler(handler)
    return SemanticActionLog(logger), handler


def _payload(message):
    assert message.startswith("ACTION ")
    return json.loads(message.removeprefix("ACTION "))


def test_immediate_actions_are_deterministic_and_monotonic():
    actions, output = _recorder()

    assert actions.record("open", z=2, a=True) == 1
    assert actions.record("export", format="png") == 2

    assert output.messages[0] == (
        'ACTION {"action":"open","action_id":1,"fields":{"a":true,"z":2}}'
    )
    assert _payload(output.messages[1])["action_id"] == 2


def test_gesture_updates_coalesce_and_cancel_is_silent():
    actions, output = _recorder()

    token = actions.begin("slider_change", control="contrast", before=10)
    assert actions.update(token, after=20)
    assert actions.update(token, after=70)
    assert output.messages == []
    assert actions.end(token, status="ok") == 1
    assert actions.end(token) is None

    cancelled = actions.begin("brush_stroke", points=[(1, 2)])
    assert actions.cancel(cancelled)
    assert len(output.messages) == 1
    assert _payload(output.messages[0])["fields"] == {
        "after": 70, "before": 10, "control": "contrast", "status": "ok"
    }


def test_sensitive_values_and_absolute_paths_are_summarized():
    actions, output = _recorder()

    actions.record(
        "save",
        path=r"C:\Users\Arsha\private\photo.secret.png",
        pixels=[1, 2, 3],
        preset_payload={"api_key": "secret", "curve": [1, 2]},
        relative_name="exports/result.png",
    )

    message = output.messages[0]
    assert "Users" not in message
    assert "api_key" not in message
    assert "[1,2,3]" not in message
    fields = _payload(message)["fields"]
    assert fields["path"] == {
        "basename": "photo.secret.png", "extension": ".png", "redacted": "path"
    }
    assert fields["pixels"] == {"length": 3, "redacted": "list"}
    assert fields["preset_payload"] == {"length": 2, "redacted": "dict"}
    assert fields["relative_name"] == "exports/result.png"


def test_strings_and_field_counts_are_bounded():
    actions, output = _recorder()
    fields = {f"k{i:02d}": "x" * 1000 for i in range(40)}

    actions.record("x" * 1000, **fields)

    payload = _payload(output.messages[0])
    assert len(payload["action"]) == 259
    assert len(payload["fields"]) == 33
    assert payload["fields"]["_omitted_fields"] == 8
    assert all(
        len(value) == 259
        for key, value in payload["fields"].items()
        if key != "_omitted_fields"
    )


def test_concurrent_records_have_unique_log_order_ids():
    actions, output = _recorder()
    threads = [
        threading.Thread(target=actions.record, args=("worker",), kwargs={"worker": i})
        for i in range(40)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    ids = [_payload(message)["action_id"] for message in output.messages]
    assert ids == list(range(1, 41))
