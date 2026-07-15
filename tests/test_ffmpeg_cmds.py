from ditherzam.video.ffmpeg import (
    cmd_probe_fps, cmd_probe_duration, cmd_probe_has_audio,
    cmd_extract_frames, cmd_encode, cmd_extract_audio,
    cmd_extract_audio_reencode, cmd_mux, parse_fps,
)


def test_all_builders_return_list_of_str():
    for c in (
        cmd_probe_fps("in.mp4"),
        cmd_probe_duration("in.mp4"),
        cmd_probe_has_audio("in.mp4"),
        cmd_extract_frames("in.mp4", "/frames"),
        cmd_encode("/frames", 30, "out.mp4"),
        cmd_extract_audio("in.mp4", "a.m4a"),
        cmd_extract_audio_reencode("in.mp4", "a.aac"),
        cmd_mux("v.mp4", "a.m4a", "out.mp4"),
    ):
        assert isinstance(c, list) and all(isinstance(x, str) for x in c)


def test_probe_fps_cmd():
    c = cmd_probe_fps("in.mp4")
    assert "ffprobe" in c[0]
    assert "-select_streams" in c and "v:0" in c
    assert "stream=r_frame_rate" in c
    assert "default=noprint_wrappers=1:nokey=1" in c
    assert c[-1] == "in.mp4"


def test_probe_duration_cmd():
    c = cmd_probe_duration("in.mp4")
    assert "format=duration" in c
    assert c[-1] == "in.mp4"


def test_probe_has_audio_cmd():
    c = cmd_probe_has_audio("in.mp4")
    assert "-select_streams" in c and "a" in c
    assert "stream=codec_type" in c


def test_extract_cmd_uses_qscale_and_pattern():
    c = cmd_extract_frames("in.mp4", "/frames")
    assert "-qscale:v" in c and "2" in c
    assert "-i" in c and "in.mp4" in c
    assert any(a.endswith("frame%06d.png") for a in c)


def test_encode_cmd_libx264_yuv420p_framerate():
    c = cmd_encode("/frames", 30, "out.mp4")
    assert "libx264" in c and "yuv420p" in c
    assert "-framerate" in c and "30" in c
    assert any(a.endswith("frame%06d.png") for a in c)
    assert c[-1] == "out.mp4"


def test_extract_audio_copy_then_reencode():
    c1 = cmd_extract_audio("in.mp4", "a.m4a")
    assert "-vn" in c1 and "-acodec" in c1 and "copy" in c1
    c2 = cmd_extract_audio_reencode("in.mp4", "a.aac")
    assert "-vn" in c2 and "aac" in c2 and "-f" in c2 and "adts" in c2


def test_mux_cmd_copy_shortest():
    c = cmd_mux("v.mp4", "a.m4a", "out.mp4")
    assert "-c" in c and "copy" in c and "-shortest" in c
    assert "v.mp4" in c and "a.m4a" in c and c[-1] == "out.mp4"


def test_parse_fps_ratio():
    assert abs(parse_fps("30000/1001") - 29.97) < 0.01
    assert parse_fps("25/1") == 25.0
    assert parse_fps("60/1") == 60.0


def test_parse_fps_plain_number():
    assert parse_fps("24") == 24.0
    assert abs(parse_fps("23.976") - 23.976) < 1e-6


def test_parse_fps_bad_input_returns_zero():
    assert parse_fps("") == 0.0
    assert parse_fps("0/0") == 0.0
    assert parse_fps("garbage") == 0.0
