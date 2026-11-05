"""Thor's remote requests and the external hold, without DDS or motors."""

import json
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from command_source import RosUdpNavCommandSource  # noqa: E402
from locomotion_mode import LocomotionModeSwitch  # noqa: E402
from remote_mode import RemoteModeServer, parse_request  # noqa: E402

THOR = ("127.0.0.1", 40000)


def request(seq, action, session="demo"):
    return json.dumps({"type": "dreamwaq_request", "session": session, "seq": seq,
                       "action": action}).encode()


def free_port():
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.server = RemoteModeServer("127.0.0.1", free_port(), "127.0.0.1",
                                       clock_ns=lambda: self.now)
        self.addCleanup(self.server.close)

    def acks(self):
        with self.server._lock:
            sent = [message for message, _ in self.server._outbox
                    if message.get("type") == "dreamwaq_ack"]
            self.server._outbox.clear()
        return [(ack["seq"], ack["action"], ack["result"]) for ack in sent]

    def test_parse_accepts_only_well_formed_requests(self):
        self.assertEqual(parse_request(request(3, "walking")), ("demo", 3, "walking"))
        for bad in (b"{", b"[]", json.dumps({"type": "nav", "session": "s", "seq": 1,
                                              "action": "walking"}).encode(),
                    request(-1, "walking"), request(True, "walking"),
                    json.dumps({"type": "dreamwaq_request", "session": "", "seq": 1,
                                "action": "walking"}).encode()):
            self.assertIsNone(parse_request(bad), bad)

    def test_only_the_allowed_host_is_heard(self):
        self.assertEqual(self.server.handle_datagram(request(1, "walking"), ("10.0.0.9", 1)),
                         "disallowed")
        self.assertEqual(self.server.pending(), [])

    def test_each_request_is_answered_once_and_repeats_get_the_same_answer(self):
        self.assertEqual(self.server.handle_datagram(request(1, "walking"), THOR), "queued")
        # Retried before the loop took it: nothing is queued twice.
        self.assertEqual(self.server.handle_datagram(request(1, "walking"), THOR), "waiting")
        (taken,) = self.server.pending()
        self.server.resolve(taken, "applied")
        self.assertEqual(self.acks(), [(1, "walking", "applied")])
        self.assertEqual(self.server.handle_datagram(request(1, "walking"), THOR), "repeat")
        self.assertEqual(self.acks(), [(1, "walking", "applied")])
        self.assertEqual(self.server.pending(), [])
        self.assertEqual(self.server.handle_datagram(request(0, "standing"), THOR), "old")

    def test_status_and_unknown_actions_are_answered_without_the_loop(self):
        self.assertEqual(self.server.handle_datagram(request(1, "status"), THOR), "status")
        self.assertEqual(self.server.handle_datagram(request(2, "jump"), THOR), "unknown")
        self.assertEqual(self.acks(), [(1, "status", "applied"), (2, "jump", "refused")])
        self.assertEqual(self.server.pending(), [])

    def test_a_request_the_loop_did_not_take_in_time_expires_instead_of_applying_late(self):
        self.server.handle_datagram(request(1, "walking"), THOR)
        self.now += int(0.6e9)
        self.assertEqual(self.server.pending(), [])
        self.assertEqual(self.acks(), [(1, "walking", "expired")])

    def test_a_new_session_supersedes_what_the_old_one_queued(self):
        self.server.handle_datagram(request(5, "walking", session="old"), THOR)
        self.server.handle_datagram(request(0, "standing", session="new"), THOR)
        self.assertEqual([r.action for r in self.server.pending()], ["standing"])
        self.assertEqual(self.acks(), [(5, "walking", "expired")])


class SocketTests(unittest.TestCase):
    def test_acks_and_status_reach_the_requester_over_udp(self):
        port = free_port()
        server = RemoteModeServer("127.0.0.1", port, "127.0.0.1", status_hz=50.0)
        self.addCleanup(server.close)
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.bind(("127.0.0.1", 0))
        client.settimeout(1.0)
        self.addCleanup(client.close)
        server.publish_status({"mode": "standing", "command": [0.0, float("nan"), 0.0]})
        client.sendto(request(1, "walking"), ("127.0.0.1", port))
        deadline = time.monotonic() + 1.0
        while not (taken := server.pending()) and time.monotonic() < deadline:
            time.sleep(0.005)
        server.resolve(taken[0], "applied", "ok")
        seen = {}
        while len(seen) < 2:
            message = json.loads(client.recv(4096))
            seen[message["type"]] = message
        self.assertEqual(seen["dreamwaq_ack"]["result"], "applied")
        self.assertEqual(seen["dreamwaq_status"]["mode"], "standing")
        # JSON has no NaN; the status says null instead of failing to send.
        self.assertEqual(seen["dreamwaq_status"]["command"], [0.0, None, 0.0])


class ExternalHoldTests(unittest.TestCase):
    def setUp(self):
        self.port = free_port()
        config = SimpleNamespace(
            max_cmd=np.array([0.95, 0.75, 0.75]), min_cmd=np.array([-0.8, -0.75, -0.75]),
            cmd_deadzone=0.1, cmd_udp_timeout=0.3, cmd_udp_bind_host="127.0.0.1",
            cmd_udp_port=self.port, cmd_udp_allowed_host="127.0.0.1",
        )
        self.source = RosUdpNavCommandSource(config)
        self.addCleanup(self.source.close)
        self.bridge = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(self.bridge.close)
        self.seq = 0
        self.still = SimpleNamespace(lx=0.0, ly=0.0, rx=0.0)

    def send(self, state="autonomy_active", reason="active", vx=0.3):
        self.seq += 1
        self.bridge.sendto(json.dumps(dict(vx=vx, vy=0.0, wz=0.0, seq=self.seq, session="b",
                                           state=state, reason=reason)).encode(),
                           ("127.0.0.1", self.port))
        deadline = time.monotonic() + 1.0
        while self.source._last_seq != self.seq and time.monotonic() < deadline:
            time.sleep(0.005)

    def test_the_bridge_reason_reaches_the_status(self):
        self.assertEqual(self.source.external_status()["state"], None)
        self.send(state="hold", reason="operator_hold", vx=0.0)
        status = self.source.external_status()
        self.assertEqual((status["state"], status["reason"]), ("hold", "operator_hold"))
        self.assertLess(status["age_s"], 1.0)

    def test_the_hold_zeroes_the_bridge_but_not_the_sticks(self):
        self.send()
        np.testing.assert_allclose(self.source.get_command(self.still), [0.3, 0.0, 0.0], atol=1e-6)
        self.assertTrue(self.source.hold_external())
        self.assertFalse(self.source.hold_external())
        self.send()
        np.testing.assert_allclose(self.source.get_command(self.still), [0.0, 0.0, 0.0])
        pushed = SimpleNamespace(lx=0.0, ly=1.0, rx=0.0)
        np.testing.assert_allclose(self.source.get_command(pushed), [0.95, 0.0, 0.0], atol=1e-6)
        # A latched bridge damping still wins over everything.
        self.send(state="damping", reason="damping", vx=0.0)
        self.assertTrue(self.source.damping_requested())
        np.testing.assert_allclose(self.source.get_command(pushed), [0.0, 0.0, 0.0])

    def test_a_released_hold_lets_the_bridge_through_and_latches_again(self):
        self.assertFalse(self.source.release_external())
        self.assertTrue(self.source.hold_external())
        self.send()
        np.testing.assert_allclose(self.source.get_command(self.still), [0.0, 0.0, 0.0])
        self.assertTrue(self.source.release_external())
        self.assertFalse(self.source.release_external())
        self.send()
        np.testing.assert_allclose(self.source.get_command(self.still), [0.3, 0.0, 0.0], atol=1e-6)
        # The next run's switch needs it latched again.
        self.assertTrue(self.source.hold_external())
        np.testing.assert_allclose(self.source.get_command(self.still), [0.0, 0.0, 0.0])


class ModeTests(unittest.TestCase):
    def test_set_mode_is_idempotent_and_leaves_the_x_edge_alone(self):
        switch = LocomotionModeSwitch(enabled=True)
        self.assertTrue(switch.set_mode("walking"))
        self.assertFalse(switch.set_mode("walking"))
        effective, mode, changed = switch.update(0, np.array([0.3, 0.0, 0.0]))
        self.assertEqual((mode, changed), (1, False))
        np.testing.assert_allclose(effective, [0.3, 0.0, 0.0], atol=1e-6)
        self.assertTrue(switch.set_mode("standing"))
        with self.assertRaises(ValueError):
            switch.set_mode("running")
        with self.assertRaises(RuntimeError):
            LocomotionModeSwitch(enabled=False).set_mode("walking")


if __name__ == "__main__":
    unittest.main()
