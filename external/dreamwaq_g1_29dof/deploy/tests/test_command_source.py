from pathlib import Path
import json
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np


DEPLOY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY))

from command_source import (  # noqa: E402
    CommandSource,
    DdsNavCommandSource,
    UdpNavCommandSource,
    configure_command_limits,
)


def _config():
    return SimpleNamespace(
        max_cmd=np.array([0.6, 0.4, 0.5]),
        cmd_deadzone=0.1,
    )


def _controller(*, lx=0.0, ly=0.0, rx=0.0):
    return SimpleNamespace(lx=lx, ly=ly, rx=rx)


def _dds_source():
    source = DdsNavCommandSource.__new__(DdsNavCommandSource)
    source.sticks = CommandSource(_config())
    source.max_cmd = np.asarray(_config().max_cmd, dtype=np.float32)
    source.min_cmd = -source.max_cmd
    source.timeout = 0.3
    source._lock = threading.Lock()
    source._latest_cmd = np.array([0.2, -0.1, 0.3], dtype=np.float32)
    source._latest_time = time.monotonic()
    source._last_seq = None
    source._last_bad_warn = 0.0
    source._last_gap_warn = 0.0
    return source


def test_dds_command_is_clipped_and_expires_to_zero():
    source = _dds_source()
    command, state = source._parse(
        '{"vx":0.8,"vy":-0.7,"wz":0.9,"seq":1,"state":"human_active"}'
    )
    np.testing.assert_allclose(command, [0.6, -0.4, 0.5])
    assert state == "human_active"

    np.testing.assert_allclose(
        source.get_command(_controller()),
        [0.2, -0.1, 0.3],
    )
    source._latest_time = time.monotonic() - 1.0
    np.testing.assert_allclose(source.get_command(_controller()), np.zeros(3))


def test_physical_wireless_stick_overrides_fresh_dds_command():
    source = _dds_source()
    expected = source.sticks.get_command(_controller(ly=0.8))
    np.testing.assert_allclose(
        source.get_command(_controller(ly=0.8)),
        expected,
    )


def _udp_source():
    source = UdpNavCommandSource.__new__(UdpNavCommandSource)
    source.sticks = CommandSource(_config())
    source.max_cmd = np.asarray(_config().max_cmd, dtype=np.float32)
    source.min_cmd = -source.max_cmd
    source.timeout = 0.3
    source._lock = threading.Lock()
    source._latest_cmd = np.zeros(3, dtype=np.float32)
    source._latest_time = 0.0
    source._last_session = None
    source._last_seq = -1
    source._retired_sessions = set()
    source._last_bad_warn = 0.0
    source._last_gap_warn = 0.0
    return source


def test_udp_command_clips_rejects_replay_and_expires_to_zero():
    source = _udp_source()
    command, state = source._parse(
        b'{"vx":0.8,"vy":-0.7,"wz":0.9,"seq":1,'
        b'"session":"thor-a","state":"human_active"}'
    )
    np.testing.assert_allclose(command, [0.6, -0.4, 0.5])
    assert state == "human_active"
    assert source._parse(
        b'{"vx":0.1,"vy":0.0,"wz":0.0,"seq":1,"session":"thor-a"}'
    ) is None

    source._latest_cmd = command
    source._latest_time = time.monotonic()
    np.testing.assert_allclose(source.get_command(_controller()), command)
    source._latest_time = time.monotonic() - 1.0
    np.testing.assert_allclose(source.get_command(_controller()), np.zeros(3))


def test_physical_wireless_stick_overrides_fresh_udp_command():
    source = _udp_source()
    source._latest_cmd = np.array([0.2, -0.1, 0.3], dtype=np.float32)
    source._latest_time = time.monotonic()
    expected = source.sticks.get_command(_controller(ly=0.8))
    np.testing.assert_allclose(source.get_command(_controller(ly=0.8)), expected)


def _export_limits(name, overrides=(0.95, 0.75, 0.75)):
    path = DEPLOY.parent / "checkpoints" / name / "exported/manifest.json"
    bounds = json.loads(path.read_text())["command_range"]
    config = _config()
    configure_command_limits(config, bounds["lower"], bounds["upper"], overrides)
    return config


def test_new3_4_uses_full_forward_range_and_scales_reverse_separately():
    config = _export_limits("new3_4_widerange_quality_390000")
    sticks = CommandSource(config)
    # Full diagonal travel exercises all signs as well as asymmetric vx caps.
    np.testing.assert_allclose(sticks.get_command(_controller(ly=1, lx=-1, rx=-1)),
                               [0.95, 0.75, 0.75])
    np.testing.assert_allclose(sticks.get_command(_controller(ly=-1, lx=1, rx=1)),
                               [-0.8, -0.75, -0.75])
    # 55% travel is halfway through the active range after the 10% deadzone.
    np.testing.assert_allclose(sticks.get_command(_controller(ly=0.55, lx=-0.55, rx=-0.55)),
                               [0.475, 0.375, 0.375])
    np.testing.assert_allclose(sticks.get_command(_controller(ly=-0.55)), [-0.4, 0, 0])
    np.testing.assert_allclose(sticks.get_command(_controller(ly=0.1, lx=-0.1, rx=0.09)),
                               [0, 0, 0])


def test_symmetric_export_keeps_its_own_command_bounds_for_the_same_flags():
    # Exercise the narrow-envelope contract without requiring a retired export.
    config = _config()
    configure_command_limits(config, [-0.4]*3, [0.4]*3, (0.95, 0.75, 0.75))
    sticks = CommandSource(config)
    np.testing.assert_allclose(sticks.get_command(_controller(ly=1, lx=-1, rx=-1)), [0.4]*3)
    np.testing.assert_allclose(sticks.get_command(_controller(ly=-1, lx=1, rx=1)), [-0.4]*3)


def test_optional_and_zero_caps_preserve_default_and_disabled_axes():
    config = _export_limits("new3_4_widerange_quality_390000", (None, None, None))
    np.testing.assert_allclose(config.max_cmd, [0.6, 0.4, 0.5])
    np.testing.assert_allclose(config.min_cmd, [-0.6, -0.4, -0.5])
    config = _export_limits("new3_4_widerange_quality_390000", (0.95, 0.0, None))
    sticks = CommandSource(config)
    np.testing.assert_allclose(sticks.get_command(_controller(ly=1, lx=-1, rx=-1)), [0.95, 0, 0.5])
    np.testing.assert_allclose(sticks.get_command(_controller(ly=-1, lx=1, rx=1)), [-0.8, 0, -0.5])


def test_external_velocity_sources_clip_to_the_same_signed_ranges():
    config = _export_limits("new3_4_widerange_quality_390000")
    for source in (_dds_source(), _udp_source()):
        source.max_cmd, source.min_cmd = config.max_cmd, config.min_cmd
        source.sticks = CommandSource(config)
        for seq, speed, expected in ((1, 2, [0.95, 0.75, 0.75]),
                                     (2, -2, [-0.8, -0.75, -0.75])):
            payload = json.dumps(dict(vx=speed, vy=speed, wz=speed, seq=seq, session="test"))
            command, _ = source._parse(payload)
            np.testing.assert_allclose(command, expected)
