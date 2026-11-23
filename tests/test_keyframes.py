"""Step 2 keyframe selection: once the motion condition is met, the sharpest frame within +-window is taken."""
from actspace_build.step2_keyframes import select_keyframes


def _frames(lapvars, dt=0.1, speed=1.0):
    return [{"timestamp": i * dt, "position": [i * dt * speed, 0.0, 0.0], "yaw_deg": 0.0, "blur_lapvar": v}
            for i, v in enumerate(lapvars)]


def test_picks_sharpest_frame_in_window():
    # one frame per 0.1 s at 1 m/s: frame 3 (0.3 m) first meets the 0.3 m condition; frame 6 is the sharpest within the +-0.5 s window
    lap = [100, 10, 10, 10, 10, 10, 500, 10, 10, 10, 10, 10, 10]
    kf = select_keyframes(_frames(lap), 0.3, 15, 0.5)
    assert kf[:2] == [0, 6]


def test_window_never_goes_before_previous_keyframe():
    lap = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 999]
    kf = select_keyframes(_frames(lap), 0.3, 15, 0.5)
    assert kf == sorted(set(kf)) and kf[0] == 0


def test_without_window_first_trigger_frame():
    lap = [1] * 10
    kf = select_keyframes(_frames(lap), 0.25, 15, 0.0)  # threshold not on a floating-point boundary
    assert kf == [0, 3, 6, 9]


def test_only_depth_matched_frames_are_keyframes():
    # frame 6 is the sharpest but has no matching depth -> take the sharpest frame with depth in the window (frame 4)
    lap = [100, 10, 10, 10, 50, 10, 500, 10, 10, 10, 10, 10, 10]
    ok = [True, False, False, False, True, False, False, True, False, False, True, False, True]
    kf = select_keyframes(_frames(lap), 0.3, 15, 0.5, ok)
    assert kf[:2] == [0, 4] and all(ok[k] for k in kf)
