import time
import unittest
from unittest.mock import patch

import zmq

from teleimager.clock_sync import ClockCalibration, calibrate_camera_clock
from teleimager.image_client import ZMQ_Responser


class ClockSyncTests(unittest.TestCase):
    def test_monotonic_mapping_ignores_robot_wall_clock(self):
        clock = ClockCalibration(1_790_000_000_000_000, 14_000_000_000_000, 5_000_000_000, 600_001)
        self.assertEqual(clock.capture_wall_time_us(6_000_000_000), 1_790_000_001_000_000)
        self.assertEqual(clock.uncertainty_us, 301)
        self.assertEqual(clock.metadata()['server_wall_time_us'], 14_000_000_000_000)

    def test_clock_rpc_preserves_camera_config_protocol(self):
        config = {'head_camera': {'binocular': False}}
        server = ZMQ_Responser(config, host='127.0.0.1', port=0)
        port = int(server._socket.getsockopt_string(zmq.LAST_ENDPOINT).rsplit(':', 1)[1])
        context = zmq.Context()
        request = context.socket(zmq.REQ)
        request.setsockopt(zmq.LINGER, 0)
        request.setsockopt(zmq.RCVTIMEO, 2000)
        request.connect(f'tcp://127.0.0.1:{port}')
        try:
            request.send(b'GET_DATA')
            self.assertEqual(request.recv_json(), config)
            calibration = calibrate_camera_clock('127.0.0.1', port, samples=3)
            aligned = calibration.capture_wall_time_us(time.monotonic_ns())
            self.assertLess(abs(aligned - time.time_ns() // 1000), 200_000)
            request.send(b'GET_DATA')
            self.assertEqual(request.recv_json(), config)
            with patch('teleimager.image_client.clock_response', return_value=config):
                with self.assertRaisesRegex(RuntimeError, 'update Nature teleimager'):
                    calibrate_camera_clock('127.0.0.1', port, samples=1)
        finally:
            request.close()
            context.term()
            server.stop()


if __name__ == '__main__':
    unittest.main()
