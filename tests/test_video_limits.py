from ditherzam.video.ffmpeg import (
    check_video_limits, MSG_FPS, MSG_DURATION, FPS_LIMIT, DURATION_LIMIT,
)


def test_reject_high_fps():
    msg = check_video_limits(75, 10, expert=False)
    assert msg == MSG_FPS


def test_reject_long_duration():
    msg = check_video_limits(30, 120, expert=False)
    assert msg == MSG_DURATION


def test_fps_checked_before_duration_when_both_bad():
    # Both over the cap -> fps message wins (checked first).
    assert check_video_limits(90, 300, expert=False) == MSG_FPS


def test_allow_within_limits():
    assert check_video_limits(30, 30, expert=False) is None


def test_boundary_exactly_at_cap_is_allowed():
    assert check_video_limits(FPS_LIMIT, DURATION_LIMIT, expert=False) is None


def test_just_over_boundary_rejected():
    assert check_video_limits(60.01, 30, expert=False) == MSG_FPS
    assert check_video_limits(30, 60.01, expert=False) == MSG_DURATION


def test_expert_bypasses_everything():
    assert check_video_limits(120, 600, expert=True) is None
    assert check_video_limits(240, 3600, expert=True) is None
