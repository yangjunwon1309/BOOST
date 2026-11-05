"""Only known background workloads may be taken off policy/writer CPUs."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from partition_autonomy_cpus import ROLE_CPUS, partition, process_roles, real_control_pids


class CpuPartitionTests(unittest.TestCase):
    def test_shadow_guard_distinguishes_real_control_from_read_only_probes(self):
        processes = {
            10: {'args': ('/env/bin/dreamwaq-g1-29dof', '--net', 'eth0')},
            11: {'args': ('/env/bin/dreamwaq-g1-29dof', '--net', 'eth0', '--shadow')},
            12: {'args': ('python3', '/repo/deploy/run_dreamwaq.py')},
            13: {'args': ('bash', '/repo/deploy/run_dreamwaq_autonomy.sh', '--shadow')},
        }
        self.assertEqual(real_control_pids(processes), [10, 12])

    def test_camera_descendants_are_classified_without_catching_policy_workers(self):
        processes = {
            10: {'ppid': 1, 'args': ('python', '-m', 'teleimager.image_server')},
            11: {'ppid': 10, 'args': ('python', '-c', 'spawn_main(...)')},
            12: {'ppid': 11, 'args': ('python', 'capture.py')},
            20: {'ppid': 1, 'args': ('/env/bin/dreamwaq-g1-29dof', '--net', 'eth0')},
            21: {'ppid': 20, 'args': ('python', '-c', 'spawn_main(...)')},
            30: {'ppid': 1, 'args': ('/ws/nature_ws/devel/lib/url_lidar_slam/lio_node',)},
            40: {'ppid': 1, 'args': ('/opt/ros/noetic/bin/rosmaster',)},
            50: {'ppid': 1, 'args': ('python3', '-m', 'url_navigation.interface.ros.ros1.nodes.lpp_node')},
            60: {'ppid': 1, 'args': ('/env/bin/python', '/env/bin/teleimager-server')},
            61: {'ppid': 60, 'args': ('python', '-c', 'spawn_main(...)')},
            70: {'ppid': 1, 'args': ('python', '-m', 'teleimager.server_cli')},
        }
        self.assertEqual(process_roles(processes),
                         {10: 'camera', 11: 'camera', 12: 'camera', 60: 'camera', 61: 'camera', 70: 'camera', 30: 'slam',
                          40: 'background', 50: 'background'})
        for cpus in ROLE_CPUS.values():
            self.assertTrue(cpus.isdisjoint({4, 5, 6, 7}))

    def test_apply_covers_all_threads_and_report_does_not_mutate(self):
        with tempfile.TemporaryDirectory() as folder:
            proc = Path(folder)
            p = proc / '30'
            for tid in (30, 31, 32):
                (p / 'task' / str(tid)).mkdir(parents=True)
            # /proc/stat tail: state, ppid, ... starttime (index 19).
            (p / 'stat').write_text('30 (lio_node) S 1 ' + '0 ' * 17 + '123 0')
            process = {30: {'ppid': 1, 'args': ('/ws/url_lidar_slam/lio_node',), 'start': 123}}
            with patch('partition_autonomy_cpus.read_processes', return_value=process), \
                 patch('partition_autonomy_cpus.os.sched_getaffinity', return_value=set(range(8))), \
                 patch('partition_autonomy_cpus.os.sched_setaffinity') as apply:
                self.assertTrue(partition(False, proc))
                apply.assert_not_called()
                self.assertTrue(partition(True, proc))
                self.assertEqual({call.args[0] for call in apply.call_args_list}, {30, 31, 32})
                self.assertTrue(all(call.args[1] == {2, 3} for call in apply.call_args_list))


if __name__ == '__main__':
    unittest.main()
