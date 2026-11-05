import contextlib
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, call, patch

import cv2
import numpy as np
import zmq

from teleimager.image_client import (
    ImageClient,
    TeleImage,
    ZMQ_PublisherThread,
    ZMQ_Requester,
    ZMQ_Responser,
    main,
)


def camera_config(port, width):
    return {
        'head_camera': {
            'enable_zmq': True, 'enable_webrtc': False, 'zmq_port': port,
            'image_shape': [16, width], 'fps': 30, 'binocular': width == 32,
        },
        'left_wrist_camera': {'enable_zmq': False},
        'right_wrist_camera': {'enable_zmq': False},
    }


def jpeg(width, color):
    image = np.full((16, width, 3), color, dtype=np.uint8)
    success, encoded = cv2.imencode('.jpg', image)
    assert success
    return encoded.tobytes()


class MultiHostReceptionTest(unittest.TestCase):
    def test_same_ports_keep_configs_and_frames_separate_and_close_independently(self):
        context = zmq.Context()
        self.addCleanup(context.term)
        publishers = []
        stream_port = 0
        for host in ('127.0.0.1', '127.0.0.2'):
            publisher = ZMQ_PublisherThread(stream_port, host, context)
            publisher.start()
            self.addCleanup(publisher.stop)
            self.assertTrue(publisher.wait_for_start())
            stream_port = int(publisher._socket.getsockopt_string(
                zmq.LAST_ENDPOINT
            ).rsplit(':', 1)[1])
            publishers.append(publisher)

        configs = [camera_config(stream_port, width) for width in (32, 16)]
        request_port = 0
        for host, config in zip(('127.0.0.1', '127.0.0.2'), configs):
            server = ZMQ_Responser(config, host=host, port=request_port)
            self.addCleanup(server.stop)
            request_port = int(server._socket.getsockopt_string(
                zmq.LAST_ENDPOINT
            ).rsplit(':', 1)[1])

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'cam_config_client.yaml'
            cache.write_text('existing single-host config\n')
            with patch('teleimager.image_client.CONFIG_CLIENT_PATH', str(cache)), patch(
                'teleimager.image_client.get_local_config',
                side_effect=AssertionError('Must use the requested host live config'),
            ):
                clients = []
                for host, config in zip(('127.0.0.1', '127.0.0.2'), configs):
                    client = ImageClient(host, request_port, request_bgr=True,
                                         use_local_config=False)
                    self.addCleanup(client.close)
                    self.assertEqual(client.get_cam_config(), config)
                    clients.append(client)

                packets = [jpeg(32, (0, 0, 230)), jpeg(16, (230, 0, 0))]
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    for publisher, packet in zip(publishers, packets):
                        publisher.send(packet)
                    frames = [client.get_head_frame().bgr for client in clients]
                    if all(frame is not None for frame in frames):
                        break
                    time.sleep(0.02)
                self.assertTrue(all(frame is not None for frame in frames))
                self.assertEqual([frame.shape for frame in frames],
                                 [(16, 32, 3), (16, 16, 3)])
                np.testing.assert_allclose(frames[0][0, 0], (0, 0, 230), atol=2)
                np.testing.assert_allclose(frames[1][0, 0], (230, 0, 0), atol=2)
                self.assertEqual(cache.read_text(), 'existing single-host config\n')

                clients[0].close()
                updated = jpeg(16, (0, 230, 0))
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    publishers[1].send(updated)
                    frame = clients[1].get_head_frame().bgr
                    if frame is not None and frame[0, 0, 1] > 200:
                        break
                    time.sleep(0.02)
                self.assertIsNotNone(frame)
                np.testing.assert_allclose(frame[0, 0], (0, 230, 0), atol=2)

    def test_unreachable_host_does_not_reuse_other_host_cache(self):
        requester = ZMQ_Requester('127.0.0.1', 65535, use_local_config=False)
        self.addCleanup(requester.close)
        with patch.object(requester._poller, 'poll', return_value=[]), patch(
            'teleimager.image_client.get_local_config', return_value=camera_config(55555, 32)
        ) as fallback:
            self.assertIsNone(requester.request())
            fallback.assert_not_called()

    def test_single_host_still_allows_local_fallback(self):
        requester = ZMQ_Requester('127.0.0.1', 65535)
        self.addCleanup(requester.close)
        config = camera_config(55555, 32)
        with patch.object(requester._poller, 'poll', return_value=[]), patch(
            'teleimager.image_client.get_local_config', return_value=config
        ):
            self.assertEqual(requester.request(), config)


class MultiHostCliTest(unittest.TestCase):
    def test_host_names_disambiguate_windows_and_fps_and_preserve_single_host_mode(self):
        for hosts in (('127.0.0.1',), ('127.0.0.1', '192.168.123.164')):
            with self.subTest(hosts=hosts):
                clients = []
                for width in (32, 16)[:len(hosts)]:
                    client = Mock()
                    image = np.zeros((16, width, 3), dtype=np.uint8)
                    client.get_cam_config.return_value = camera_config(55555, width)
                    client.get_head_frame.return_value = TeleImage(
                        fps=30, jpg=b'jpeg', bgr=image
                    )
                    clients.append(client)
                multiple = len(hosts) > 1
                argv = ['--hosts' if multiple else '--host', *hosts]
                output = io.StringIO()
                with patch('teleimager.image_client.ImageClient', side_effect=clients) as factory, patch(
                    'teleimager.image_client.time.monotonic', side_effect=(0, 1)
                ), patch('teleimager.image_client.cv2.imshow') as show, patch(
                    'teleimager.image_client.cv2.waitKey', return_value=ord('q')
                ), patch('teleimager.image_client.cv2.destroyAllWindows'), contextlib.redirect_stdout(output):
                    main(argv)
                self.assertEqual([item.args[0] for item in show.call_args_list],
                                 [f'{host}/head_camera' if multiple else 'head_camera'
                                  for host in hosts])
                self.assertEqual(factory.call_args_list, [
                    call(host=host, request_bgr=True, use_local_config=not multiple)
                    for host in hosts
                ])
                for host, client in zip(hosts, clients):
                    client.close.assert_called_once()
                    prefix = f'[{host}] ' if multiple else ''
                    self.assertIn(prefix + '[    1.0s]', output.getvalue())

    def test_connection_failure_identifies_host_and_closes_previous_client(self):
        client = Mock()
        client.get_cam_config.return_value = camera_config(55555, 32)
        error = io.StringIO()
        with patch('teleimager.image_client.ImageClient', side_effect=[
            client, RuntimeError('Failed to get camera configuration.')
        ]), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
            with self.assertRaises(SystemExit) as exited:
                main(['--hosts', '127.0.0.1', '192.168.123.164', '--no-display'])
        self.assertEqual(exited.exception.code, 1)
        self.assertIn('192.168.123.164', error.getvalue())
        client.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
