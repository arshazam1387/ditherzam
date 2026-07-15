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


from ditherzam.video.ffmpeg import probe_fps, probe_duration, probe_has_audio


def test_probe_fps_parses_runner_output():
    fake = lambda cmd: "30000/1001\n"
    assert abs(probe_fps("in.mp4", runner=fake) - 29.97) < 0.01


def test_probe_duration_parses_runner_output():
    fake = lambda cmd: "12.480000\n"
    assert abs(probe_duration("in.mp4", runner=fake) - 12.48) < 1e-6


def test_probe_duration_bad_output_is_zero():
    fake = lambda cmd: "N/A\n"
    assert probe_duration("in.mp4", runner=fake) == 0.0


def test_probe_has_audio_true_when_codec_type_present():
    assert probe_has_audio("in.mp4", runner=lambda cmd: "audio\n") is True


def test_probe_has_audio_false_when_empty():
    assert probe_has_audio("in.mp4", runner=lambda cmd: "\n") is False
