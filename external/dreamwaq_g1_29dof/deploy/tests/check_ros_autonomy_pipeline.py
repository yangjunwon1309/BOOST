#!/usr/bin/env python3
"""Real ROS1 + UDP integration on a private master; never opens robot DDS."""

import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import xmlrpc.client

import numpy as np

DEPLOY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY))
from command_source import RosUdpNavCommandSource, check_command_damping


def free_port(kind=socket.SOCK_STREAM):
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def until(predicate, publish=lambda: None, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        publish()
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError('condition timed out')


def stop_process(process, group=False):
    if process.poll() is not None:
        return
    if group:
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if group:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=5)


def main():
    master_port, udp_port = free_port(), free_port(socket.SOCK_DGRAM)
    env = dict(os.environ, ROS_MASTER_URI='http://127.0.0.1:%d' % master_port,
               ROS_IP='127.0.0.1')
    env.pop('ROS_HOSTNAME', None)
    os.environ.update(env)
    os.environ.pop('ROS_HOSTNAME', None)
    source = None
    bridge = None
    with tempfile.TemporaryFile() as log:
        master = subprocess.Popen(['roscore', '-p', str(master_port)], env=env,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            def ready():
                try:
                    return xmlrpc.client.ServerProxy(env['ROS_MASTER_URI']).getPid('/test')[0] == 1
                except (OSError, xmlrpc.client.Error):
                    return False
            until(ready, timeout=15)
            import rospy
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from std_msgs.msg import Bool, Empty, String
            rospy.init_node('dreamwaq_autonomy_motor_free_test', disable_signals=True)
            config = SimpleNamespace(cmd_source='ros_udp', max_cmd=np.array([0.95, 0.75, 0.75]),
                min_cmd=np.array([-0.8, -0.75, -0.75]), cmd_deadzone=0.1,
                cmd_udp_timeout=0.3, cmd_udp_bind_host='127.0.0.1',
                cmd_udp_port=udp_port, cmd_udp_allowed_host='127.0.0.1')
            source = RosUdpNavCommandSource(config)
            neutral = SimpleNamespace(lx=0, ly=0, rx=0)
            stick = SimpleNamespace(lx=0, ly=1, rx=0)
            bridge = subprocess.Popen(['bash', str(DEPLOY / 'run_ros_autonomy_bridge.sh'),
                '--no-keyboard', '--port', str(udp_port)], env=env, stdout=log, stderr=subprocess.STDOUT)
            cmd_pub = rospy.Publisher('/cmd/velocity', Twist, queue_size=1)
            odom_pub = rospy.Publisher('/dreamstep_odom_g1', Odometry, queue_size=1)
            hold_pub = rospy.Publisher('/dreamwaq/hold', Bool, queue_size=1)
            damp_pub = rospy.Publisher('/dreamwaq/damping', Empty, queue_size=1)
            align_pub = rospy.Publisher('/align/state', String, queue_size=1)
            pubs = [cmd_pub, odom_pub, hold_pub, damp_pub, align_pub]
            until(lambda: all(pub.get_num_connections() > 0 for pub in pubs))
            until(lambda: source._last_seq >= 0)
            command = Twist()
            command.linear.x, command.linear.y, command.angular.z = 2.0, -2.0, 2.0
            odometry = Odometry()
            odometry.pose.pose.orientation.w = 1
            def fresh():
                cmd_pub.publish(command)
                odom_pub.publish(odometry)
            fresh()
            time.sleep(0.1)
            np.testing.assert_allclose(source.get_command(neutral), [0, 0, 0])
            np.testing.assert_allclose(source.get_command(stick), [0.95, 0, 0])
            print('PASS startup HOLD stops autonomy and preserves manual control')
            hold_pub.publish(Bool(data=False))
            until(lambda: np.allclose(source.get_command(neutral), [0.95, -0.75, 0.75]), fresh)
            np.testing.assert_allclose(source.get_command(stick), [0.95, 0, 0])
            print('PASS ROS Twist -> bridge -> UDP -> signed clipping and stick override')
            until(lambda: np.allclose(source.get_command(neutral), [0, 0, 0]),
                  lambda: odom_pub.publish(odometry))
            print('PASS planner timeout zeros velocity while ROS bridge heartbeat continues')
            until(lambda: source.get_command(neutral)[0] > 0, fresh)
            until(lambda: source._latest_state == 'hold', lambda: cmd_pub.publish(command))
            np.testing.assert_allclose(source.get_command(neutral), [0, 0, 0])
            np.testing.assert_allclose(source.get_command(stick), [0.95, 0, 0])
            print('PASS odometry timeout stops autonomy and preserves manual control')
            until(lambda: source.get_command(neutral)[0] > 0, fresh)
            align_pub.publish(String(data='ALIGNED'))
            until(lambda: source._latest_state == 'hold', fresh)
            np.testing.assert_allclose(source.get_command(neutral), [0, 0, 0])
            np.testing.assert_allclose(source.get_command(stick), [0.95, 0, 0])
            print('PASS ALIGNED enters HOLD')
            hold_pub.publish(Bool(data=False))
            until(lambda: source.get_command(neutral)[0] > 0, fresh)
            bridge.terminate()
            bridge.wait(timeout=5)
            bridge = None
            # The shutdown packet requests HOLD; loss of the packet is also
            # covered by the receiver's independent 0.3 s watchdog.
            time.sleep(0.4)
            np.testing.assert_allclose(source.get_command(neutral), [0, 0, 0])
            np.testing.assert_allclose(source.get_command(stick), [0.95, 0, 0])
            print('PASS bridge shutdown/loss stops autonomy and preserves manual control')
            bridge = subprocess.Popen(['bash', str(DEPLOY / 'run_ros_autonomy_bridge.sh'),
                '--no-keyboard', '--port', str(udp_port)], env=env, stdout=log, stderr=subprocess.STDOUT)
            until(lambda: damp_pub.get_num_connections() > 0)
            until(source.damping_requested, lambda: damp_pub.publish(Empty()))
            np.testing.assert_allclose(source.get_command(stick), [0, 0, 0])
            try:
                check_command_damping(source)
                raise AssertionError('damping cleanup hook did not fire')
            except KeyboardInterrupt:
                pass
            print('PASS bridge restart + ROS damping latches runtime cleanup request')
            print('PASS private ROS master only; no Unitree SDK or LowCmd publisher created')
        except BaseException:
            log.seek(0)
            print(log.read().decode('utf-8', errors='replace'), file=sys.stderr)
            raise
        finally:
            if source is not None:
                source.close()
            if bridge is not None and bridge.poll() is None:
                stop_process(bridge)
            if 'rospy' in sys.modules:
                sys.modules['rospy'].signal_shutdown('test complete')
            stop_process(master, group=True)


if __name__ == '__main__':
    main()
