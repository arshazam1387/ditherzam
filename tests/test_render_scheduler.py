"""Qt-free tests for the priority-aware render scheduler."""
from ditherzam.render import RenderSettings
from ditherzam.ui.render_request import RenderKind, RenderRequest
from ditherzam.ui.render_scheduler import RenderScheduler


def _req(kind=RenderKind.DRAG, source_id=1, target_max_side=640,
         logical_size=(100, 100)):
    # generation is a placeholder here -- the scheduler stamps the real value.
    return RenderRequest(
        generation=0, kind=kind, settings=RenderSettings(), source_id=source_id,
        target_max_side=target_max_side, logical_size=logical_size,
    )


def test_first_request_starts_and_returns_stamped_request():
    s = RenderScheduler()
    req = s.request(_req())
    assert req is not None
    assert req.generation == 1
    assert s.is_current(req)


def test_request_while_busy_is_coalesced():
    s = RenderScheduler()
    assert s.request(_req()) is not None       # in flight
    assert s.request(_req()) is None            # coalesced -> pending
    assert s.request(_req()) is None            # still pending, no new worker


def test_finish_without_pending_starts_nothing():
    s = RenderScheduler()
    s.request(_req())
    assert s.on_finished() is None
    # a fresh request after idle starts again with the next generation
    nxt = s.request(_req())
    assert nxt is not None and nxt.generation == 2


def test_finish_with_pending_starts_trailing_render():
    s = RenderScheduler()
    first = s.request(_req())              # generation 1 in flight
    s.request(_req())                      # pending
    nxt = s.on_finished()                  # first done -> trailing
    assert nxt is not None
    assert nxt.generation == 2
    assert s.is_current(nxt)
    assert not s.is_current(first)


def test_only_latest_request_is_current():
    s = RenderScheduler()
    t1 = s.request(_req())
    s.request(_req())                      # pending
    t2 = s.on_finished()                   # trailing
    assert not s.is_current(t1)
    assert s.is_current(t2)


def test_drag_burst_coalesces_to_one_trailing_render():
    """20 rapid requests during one in-flight render collapse to a single
    trailing render, not 20."""
    s = RenderScheduler()
    started = 1
    s.request(_req())                      # first render starts
    for _ in range(20):
        if s.request(_req()) is not None:
            started += 1                   # would start a worker
    assert started == 1                    # nothing else started while busy
    assert s.on_finished() is not None     # first finishes -> one trailing render
    started += 1
    assert s.on_finished() is None         # trailing finishes, nothing pending
    assert started == 2


def test_invalidate_makes_inflight_stale():
    s = RenderScheduler()
    req = s.request(_req())                # background render in flight
    s.invalidate()                         # a synchronous render_now happened
    assert not s.is_current(req)           # the in-flight result must not paint


def test_invalidate_drops_pending_trailing_request():
    """A synchronous render_now() (which calls invalidate(), never request())
    must not leave a stale coalesced pending request to be promoted and painted
    over the fresh full render. Distinct from the priority path: render_now
    never calls request(), so bumping the generation alone is not enough --
    the pending request itself must be cleared."""
    s = RenderScheduler()
    s.request(_req())                      # gen 1 in flight
    s.request(_req())                      # coalesced -> pending, gen 1
    s.invalidate()                         # render_now() painted synchronously
    assert s.on_finished() is None         # stale pending must NOT be promoted


def test_out_of_order_delivery_paints_only_current():
    s = RenderScheduler()
    t1 = s.request(_req())
    s.request(_req())                      # pending
    t2 = s.on_finished()                   # trailing launched
    assert s.is_current(t2)
    assert not s.is_current(t1)            # late/stale delivery must not paint


# ---- priority tests -------------------------------------------------------

def test_higher_generation_supersedes_pending_regardless_of_kind():
    """invalidate() bumps the generation; the next request -- even a lowly
    drag -- must replace an already-pending higher-priority Full from the
    stale generation, since newest state always wins."""
    s = RenderScheduler()
    s.request(_req(kind=RenderKind.DRAG))          # gen 1 in flight
    s.request(_req(kind=RenderKind.FULL))          # pending, gen 1, FULL
    s.invalidate()                                 # gen bumps to 2
    s.request(_req(kind=RenderKind.DRAG))          # new gen 2 request
    nxt = s.on_finished()
    assert nxt is not None
    assert nxt.kind is RenderKind.DRAG
    assert nxt.generation == 3


def test_full_supersedes_capped_within_same_generation():
    s = RenderScheduler()
    s.request(_req(kind=RenderKind.DRAG))          # in flight
    s.request(_req(kind=RenderKind.SETTLE))        # pending: SETTLE
    s.request(_req(kind=RenderKind.FULL))          # replaces pending: FULL
    nxt = s.on_finished()
    assert nxt is not None
    assert nxt.kind is RenderKind.FULL


def test_capped_does_not_supersede_already_pending_full():
    s = RenderScheduler()
    s.request(_req(kind=RenderKind.DRAG))          # in flight
    s.request(_req(kind=RenderKind.FULL))          # pending: FULL
    s.request(_req(kind=RenderKind.SETTLE))        # must NOT replace FULL
    s.request(_req(kind=RenderKind.DRAG))          # must NOT replace FULL either
    nxt = s.on_finished()
    assert nxt is not None
    assert nxt.kind is RenderKind.FULL


def test_settle_and_zoom_outrank_drag_but_not_each_other_specially():
    s = RenderScheduler()
    s.request(_req(kind=RenderKind.DRAG))          # in flight
    s.request(_req(kind=RenderKind.ZOOM))          # pending: ZOOM
    s.request(_req(kind=RenderKind.DRAG))          # must NOT replace ZOOM
    nxt = s.on_finished()
    assert nxt is not None
    assert nxt.kind is RenderKind.ZOOM
