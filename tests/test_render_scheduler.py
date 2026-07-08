"""Qt-free tests for the render coalescing state machine."""
from ditherzam.ui.render_scheduler import RenderCoalescer


def test_first_request_starts_and_returns_token():
    c = RenderCoalescer()
    tok = c.request()
    assert tok == 1
    assert c.is_current(1)


def test_request_while_busy_is_coalesced():
    c = RenderCoalescer()
    assert c.request() == 1        # in flight
    assert c.request() is None     # coalesced -> pending
    assert c.request() is None     # still pending, no new worker


def test_finish_without_pending_starts_nothing():
    c = RenderCoalescer()
    c.request()
    assert c.on_finished() is None
    # a fresh request after idle starts again with the next token
    assert c.request() == 2


def test_finish_with_pending_starts_trailing_render():
    c = RenderCoalescer()
    c.request()                    # token 1 in flight
    c.request()                    # pending
    nxt = c.on_finished()          # token 1 done -> trailing token 2
    assert nxt == 2
    assert c.is_current(2)
    assert not c.is_current(1)


def test_only_latest_token_is_current():
    c = RenderCoalescer()
    t1 = c.request()
    c.request()                    # pending
    t2 = c.on_finished()           # trailing
    assert not c.is_current(t1)
    assert c.is_current(t2)


def test_drag_burst_coalesces_to_one_trailing_render():
    """20 rapid requests during one in-flight render collapse to a single trailing
    render, not 20."""
    c = RenderCoalescer()
    started = 1
    c.request()                    # first render starts
    for _ in range(20):
        if c.request() is not None:
            started += 1           # would start a worker
    # nothing else started while busy
    assert started == 1
    # first finishes -> exactly one trailing render
    assert c.on_finished() is not None
    started += 1
    # that trailing render finishes with no further pending -> done
    assert c.on_finished() is None
    assert started == 2


def test_invalidate_makes_inflight_stale():
    c = RenderCoalescer()
    tok = c.request()              # background render in flight
    c.invalidate()                 # a synchronous render_now happened
    assert not c.is_current(tok)   # the in-flight result must not paint


def test_out_of_order_delivery_paints_only_current():
    c = RenderCoalescer()
    t1 = c.request()
    c.request()                    # pending
    t2 = c.on_finished()           # trailing token 2 launched
    # simulate t2 delivering before... it's the current one
    assert c.is_current(t2)
    # a late/stale delivery of t1 must not be current
    assert not c.is_current(t1)
