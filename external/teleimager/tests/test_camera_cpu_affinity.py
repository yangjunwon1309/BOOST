"""Check Linux thread/child inheritance in an isolated subprocess, without cameras."""

import argparse
import json
import os
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

from teleimager.cpu_affinity import bootstrap_camera_affinity, parse_camera_cpus


class CameraAffinityTests(unittest.TestCase):
    def test_default_custom_mask_opt_out_and_help(self):
        with patch('teleimager.cpu_affinity.pin_camera_tree') as pin:
            bootstrap_camera_affinity(['--config', 'camera.yaml'])
            pin.assert_called_once_with((0, 1))
        with patch('teleimager.cpu_affinity.pin_camera_tree') as pin:
            bootstrap_camera_affinity(['--camera-cpus', '2-3', '--rs'])
            pin.assert_called_once_with((2, 3))
        for flags in (['--no-affinity'], ['--help']):
            with self.subTest(flags=flags), patch('teleimager.cpu_affinity.pin_camera_tree') as pin:
                bootstrap_camera_affinity(flags)
                pin.assert_not_called()
        for value in ('', '-1', '3-1', '0,,1', 'x'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_camera_cpus(value)

    @unittest.skipUnless(hasattr(os, 'sched_setaffinity'), 'Linux affinity required')
    def test_existing_threads_descendants_and_future_workers_use_camera_cpus(self):
        available = sorted(os.sched_getaffinity(0))
        if len(available) < 2:
            self.skipTest('requires at least two available CPUs')
        target = available[:2]
        child_program = textwrap.dedent('''
            import json, os, subprocess, sys, threading
            from pathlib import Path
            done = threading.Event()
            worker = threading.Thread(target=done.wait)
            worker.start()
            grandchild = subprocess.Popen([sys.executable, '-c',
                "import json,os,sys; print('ready',flush=True); sys.stdin.readline(); print(json.dumps(sorted(os.sched_getaffinity(0))),flush=True)"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            assert grandchild.stdout.readline().strip() == 'ready'
            print('ready', flush=True)
            sys.stdin.readline()
            grandchild_mask, _ = grandchild.communicate(input='\\n', timeout=10)
            masks = [sorted(os.sched_getaffinity(int(t.name)))
                     for t in Path('/proc/self/task').iterdir()]
            done.set()
            worker.join()
            print(json.dumps({'threads': masks, 'grandchild': json.loads(grandchild_mask)}), flush=True)
        ''')
        program = textwrap.dedent('''
            import json, os, subprocess, sys, threading
            from pathlib import Path
            from teleimager.cpu_affinity import pin_camera_tree
            target, child_program = json.loads(sys.argv[1])
            done = threading.Event()
            worker = threading.Thread(target=done.wait)
            worker.start()
            child = subprocess.Popen([sys.executable, '-c', child_program],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            try:
                assert child.stdout.readline().strip() == 'ready'
                pinned = pin_camera_tree(target)
                masks = [sorted(os.sched_getaffinity(int(t.name)))
                         for t in Path('/proc/self/task').iterdir()]
                old_child, _ = child.communicate(input='\\n', timeout=10)
                future = subprocess.check_output([sys.executable, '-c',
                    'import json,os; print(json.dumps(sorted(os.sched_getaffinity(0))))'], text=True, timeout=10)
                print(json.dumps({'threads': masks, 'old_child': json.loads(old_child),
                                  'future': json.loads(future), 'processes': len(pinned)}))
            finally:
                done.set()
                worker.join()
                if child.poll() is None:
                    child.kill()
                    child.wait()
        ''')
        before = os.sched_getaffinity(0)
        result = subprocess.run([sys.executable, '-c', program, json.dumps([target, child_program])],
                                capture_output=True, text=True, timeout=20, check=True)
        data = json.loads(result.stdout)
        self.assertGreaterEqual(len(data['threads']), 2)
        self.assertGreaterEqual(len(data['old_child']['threads']), 2)
        self.assertGreaterEqual(data['processes'], 3)
        for mask in data['threads'] + data['old_child']['threads']:
            self.assertEqual(mask, target)
        self.assertEqual(data['old_child']['grandchild'], target)
        self.assertEqual(data['future'], target)
        self.assertEqual(os.sched_getaffinity(0), before)


if __name__ == '__main__':
    unittest.main()
