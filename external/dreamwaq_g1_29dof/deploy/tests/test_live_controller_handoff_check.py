"""API capability checks must not mistake an unchanged FSM for RPC success."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_live_controller_handoff import check_live_return_api, require_dreamwaq_stopped
from controller_handoff import MotionServiceError


class LiveReturnCheckTest(unittest.TestCase):
    def service(self, code=0, fsm=500):
        return Mock(check_mode=Mock(return_value="ai"),
                    _fsm=Mock(return_value=fsm),
                    loco=Mock(SwitchToInternalCtrl=Mock(return_value=code)))

    def test_server_not_implemented_is_distinct_from_client_missing(self):
        service = self.service(3203)
        report = check_live_return_api(service, guard=Mock())
        self.assertEqual(report["status"], "server_api_not_implemented")
        self.assertEqual(report["rpc_code"], 3203)
        self.assertEqual(report["fsm_after"], 500)
        service.loco.SwitchToInternalCtrl.assert_called_once_with(2)
        service.release.assert_not_called()
        service.select.assert_not_called()
        service.loco.SwitchToUserCtrl.assert_not_called()
        service.loco.SetFsmId.assert_not_called()
        service.loco = object()
        self.assertEqual(check_live_return_api(service, guard=Mock())["status"],
                         "client_api_missing")

    def test_timeout_and_transport_error_do_not_prove_support(self):
        for code in (3102, 3104):
            with self.subTest(code=code):
                service = self.service(code)
                report = check_live_return_api(service, guard=Mock())
                self.assertEqual(report["status"], "check_failed")
                service.loco.SwitchToInternalCtrl.assert_called_once()
                service._fsm.assert_called_once()

    def test_requires_native_walking_before_sending_any_request(self):
        for fsm in (0, 1, 4, 802):
            service = self.service(fsm=fsm)
            self.assertEqual(check_live_return_api(service, guard=Mock())["status"], "check_failed")
            service.loco.SwitchToInternalCtrl.assert_not_called()
        service = self.service()
        service.check_mode.return_value = ""
        check_live_return_api(service, guard=Mock())
        service._fsm.assert_not_called()
        service.loco.SwitchToInternalCtrl.assert_not_called()

    def test_acknowledgement_requires_walking_but_does_not_validate_user_control(self):
        service = self.service()
        service._fsm.side_effect = [500, 1]
        self.assertEqual(check_live_return_api(service, guard=Mock())["status"], "check_failed")
        service._fsm.side_effect = [500, 500]
        report = check_live_return_api(service, guard=Mock())
        self.assertEqual(report["status"], "return_api_acknowledged")
        self.assertFalse(report["continuous_support_validated"])

    def test_runtime_starting_between_reads_prevents_request(self):
        service = self.service()
        guard = Mock(side_effect=[None, MotionServiceError("DreamWaQ is running")])
        self.assertEqual(check_live_return_api(service, guard)["status"], "check_failed")
        service.loco.SwitchToInternalCtrl.assert_not_called()

    def test_proc_guard_ignores_own_pid_and_handles_module_entrypoint(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = root / "101"
            entry.mkdir()
            (entry / "cmdline").write_bytes(b"python\0/path/run_dreamwaq.py\0--net\0eth0\0")
            require_dreamwaq_stopped(root, own_pid=101)
            with self.assertRaisesRegex(MotionServiceError, "PID 101"):
                require_dreamwaq_stopped(root, own_pid=102)
            (entry / "cmdline").write_bytes(
                b"python\0-m\0dreamwaq_interface.scripts.homie_base_dreamwaq\0--net\0eth0\0")
            with self.assertRaisesRegex(MotionServiceError, "PID 101"):
                require_dreamwaq_stopped(root, own_pid=102)


if __name__ == "__main__":
    unittest.main()
