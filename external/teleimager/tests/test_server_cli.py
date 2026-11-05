import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml

from teleimager.image_server import main


class ServerRotationCliTest(unittest.TestCase):
    def test_camera_defaults_pin_before_server_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'camera.yaml'
            path.write_text('head_camera: {}\n')
            for flags, expected in (([], (0, 1)), (['--camera-cpus', '0-1'], (0, 1)),
                                    (['--camera-cpus', '1,2'], (1, 2))):
                events = []
                with self.subTest(flags=flags), patch(
                    'teleimager.image_server.set_performance_mode',
                    side_effect=lambda **kw: events.append(('affinity', kw['cores']))
                ), patch('teleimager.image_server.ImageServer',
                         side_effect=lambda *a, **kw: events.append(('server',)) or Mock()), \
                     patch('teleimager.image_server.signal.signal'), \
                     patch('teleimager.image_server.time.sleep'), \
                     patch('teleimager.image_server.os.killpg'):
                    main(['--config', str(path), *flags])
                self.assertEqual(events, [('affinity', expected), ('server',)])

    def test_explicit_opt_out_does_not_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'camera.yaml'
            path.write_text('head_camera: {}\n')
            with patch('teleimager.image_server.set_performance_mode') as pin, \
                 patch('teleimager.image_server.ImageServer'), \
                 patch('teleimager.image_server.signal.signal'), \
                 patch('teleimager.image_server.time.sleep'), \
                 patch('teleimager.image_server.os.killpg'):
                main(['--no-affinity', '--config', str(path)])
            pin.assert_not_called()

    def test_cli_overrides_only_head_rotation_and_preserves_yaml(self):
        config = {
            'head_camera': {'rotation': 90},
            'left_wrist_camera': {'rotation': 180},
            'right_wrist_camera': {'rotation': 270},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'camera.yaml'
            original = yaml.safe_dump(config)
            path.write_text(original)
            for rotation in (None, 0, 90, 180, 270):
                with self.subTest(rotation=rotation), patch(
                    'teleimager.image_server.ImageServer'
                ) as server, patch('teleimager.image_server.signal.signal'), patch(
                    'teleimager.image_server.time.sleep'
                ), patch('teleimager.image_server.os.killpg'):
                    argv = ['--no-affinity', '--config', str(path)]
                    if rotation is not None:
                        argv.extend(['--rotation', str(rotation)])
                    main(argv)
                    effective = server.call_args.args[0]
                    self.assertEqual(effective['head_camera']['rotation'],
                                     90 if rotation is None else rotation)
                    self.assertEqual(effective['left_wrist_camera'], config['left_wrist_camera'])
                    self.assertEqual(effective['right_wrist_camera'], config['right_wrist_camera'])
                    self.assertEqual(path.read_text(), original)

    def test_invalid_rotation_is_rejected_before_camera_initialization(self):
        with patch('teleimager.image_server.ImageServer') as server, contextlib.redirect_stderr(
            io.StringIO()
        ):
            with self.assertRaises(SystemExit) as error:
                main(['--rotation', '45'])
        self.assertEqual(error.exception.code, 2)
        server.assert_not_called()


if __name__ == '__main__':
    unittest.main()
