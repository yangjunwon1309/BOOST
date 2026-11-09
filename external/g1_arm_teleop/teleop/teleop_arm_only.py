"""BOOST arm-only adaptation of the verified G1 teleop recorder.

Same teleoperation behaviour as teleop_hand_and_arm.py; the difference is what
reaches disk. Every addition here exists because its absence cost real work when
analysing earlier G1 teleoperation data:

  lower body      The old export carried leg joint *positions* and a base pose,
                  but no operator command, so the lower body could not be learned
                  at all -- the only base action in the file was a copy of the
                  measured odometry. Here the locomotion command is always
                  written (zeros when nothing was sent), next to the base twist
                  and projected gravity a policy needs to close that loop.

  base pose       Recorded, but deliberately *not* meant as a policy input: its
                  x/y/yaw live in a session-local odom frame that drifts, and in
                  clipped training data they behave like a clip id. They are kept
                  because segmentation, evaluation and label checks all need them.

  EE target       The pre-IK wrist target: the operator's actual intent, alive
                  for one loop iteration inside this process and recoverable from
                  nothing else. Everything else about the EE is a function of the
                  joint channels, so it is left to the conversion step, where the
                  frame convention can still be chosen (--record-ee-pose writes it
                  here anyway). Computing it there also makes the ~10 frame
                  (0.33 s) EE/joint desync of the old export unrepresentable: the
                  pose comes from the joints of that same row by construction.

  ik_residual     Target vs achieved wrist pose, per frame. In the old dataset
                  ~0.7% of frames had the arm parked while the target kept
                  moving (teleop not engaged yet), which silently became action
                  labels up to 37 cm away from what the arm did. One threshold on
                  this number removes them.

  timestamps      Every DDS sample carries when it was received. SportModeState_
                  publishes stamp=0 on G1, so receive time is the only clock
                  there is, and without it a channel offset cannot be found after
                  the fact -- which is exactly how the stale EE went unnoticed.

Topic choices are deliberate: rt/lowstate, rt/odommodestate and rt/secondary_imu,
never their rt/lf/* mirrors, which all publish at 20 Hz. The earlier dataset's
base pose was a 20 Hz signal held into 30 fps for that reason.

Usage (same keys as teleop_hand_and_arm.py: [r] track, [s] record toggle, [q] quit),
plus the number keys [1]..[9], which set the subtask label written with every frame
while recording ([0] clears it back to unlabeled). The names come from --task-steps
and the assignment is sticky, so a repeated step is simply the same digit again:

    cd teleop
    python teleop_ikea.py --record --input-mode controller --motion \
        --task-name ikea_table --task-dir ./utils/data/ \
        --task-steps "align table; grab leg; insert leg; screw leg; flip table"

In controller mode both of the operator's hands are on the controllers, so the
digits are meant for a second person watching the monitor.

Everything written is additive: left_arm/right_arm/left_ee/right_ee .qpos keep
their old meaning and length, so existing converter configs keep working.
"""

import time
import argparse
import threading
import numpy as np
import os
import sys


# Upstream caps the locomotion command at 0.3 to keep the base gentle:
# https://github.com/unitreerobotics/xr_teleoperate/issues/135
LOCO_VELOCITY_SCALE = 0.3
# G1_29_JointIndex spans 35 slots; 29..34 are unused padding kept so that
# body.qpos / body.qvel / body.torque share one index space.
BODY_MOTOR_SLOTS = 35
# Unitree G1_29 SDK order; body rows retain 35 slots, with six trailing pads.
G1_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
XR_INPUT_TIMEOUT_S = 1.0


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="G1_29 arm-only XR teleoperation and whole-body recorder.")
    # basic control parameters
    parser.add_argument('--frequency', type=float, default=30.0, help='control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='controller',
                        help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'],
                        default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29'], default='G1_29',
                        help='Only G1_29 is supported by this recorder')
    parser.add_argument('--arm-reference-mode', type=str, choices=['fixed', 'head_position', 'head_yaw'],
                        default='fixed',
                        help="Frame the wrist targets are anchored to. 'fixed' locks the anchor to the head "
                             "pose captured when tracking starts, so only your hands move the arms. "
                             "'head_position' re-anchors to the live head position every cycle; 'head_yaw' "
                             "also rotates the targets with head yaw. Press [c] to re-capture in fixed mode.")
    # network parameters
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164',
                        help='IP address of image server, used by teleimager and televuer. Every camera '
                             'not moved by --camera-host-for is read from here.')
    parser.add_argument('--camera-host-for', action='append', default=[], metavar='CAMERA=HOST',
                        help='Read one camera from another teleimager server, e.g. '
                             'right_wrist=192.168.123.164 when the right wrist is wired into the G1 PC2 '
                             'while the head and left wrist stay on Thor. CAMERA is head, left_wrist or '
                             'right_wrist (or the *_camera spelling); repeat for more cameras. Each server '
                             'is asked for its own config on the usual request port 60000, so the stream '
                             'ports stay whatever that server publishes.')
    parser.add_argument('--network-interface', type=str, default=None,
                        help='Network interface for dds communication, e.g., eth0, wlan0.')
    parser.add_argument('--head-webrtc-url', default=None,
                        help='Headset-reachable head /offer URL when the PC uses wired camera IPs.')
    parser.add_argument('--camera-clock-sync', action='store_true',
                        help='Align TIM2 capture timestamps to this PC using the server monotonic clock.')
    # mode flags
    parser.add_argument('--motion', action='store_true',
                         help='Publish arm targets on rt/arm_sdk instead of rt/lowcmd directly (needed '
                              'whenever something else -- the onboard AI/sport service, or an external '
                              'low-level controller like HOMIE -- already owns rt/lowcmd exclusively). '
                              'Does NOT by itself drive base locomotion; see --loco-client.')
    parser.add_argument('--loco-client', action='store_true',
                         help='Also drive base locomotion via the onboard LocoClient RPC service '
                              '(joystick thumbsticks -> Move()/Damp()). Requires --motion and '
                              '--input-mode controller, and requires the G1 sport/motion service to be '
                              'running. Leave this OFF when an external low-level controller (e.g. HOMIE) '
                              'owns leg/waist motion instead -- with the sport service stopped for that, '
                              'LocoClient RPCs have nothing to answer and only spam "[ClientStub] send '
                              'request error". cmd_vel is still read from the sticks and recorded either way.')
    parser.add_argument('--skip-motion-switch', action='store_true',
                         help='Skip the MotionSwitcher RPC handshake entirely (only relevant without '
                              '--motion). Use this when an external low-level controller (e.g. HOMIE via '
                              'g1_lowlevel_bridge) already owns rt/lowcmd and the onboard sport/motion '
                              'service is stopped -- that RPC has nothing to answer it and only spams '
                              '"[ClientStub] send request error" without changing behavior (it is caught '
                              'and ignored either way).')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action='store_true', help='Enable isaac simulation mode')
    parser.add_argument('--ipc', action='store_true', help='Enable IPC input instead of terminal keys')
    parser.add_argument('--affinity', action='store_true', help='Enable high priority and set CPU affinity mode')
    # record mode and task info
    parser.add_argument('--record', action='store_true', help='Enable data recording mode')
    parser.add_argument('--record-head-eye', choices=['both', 'left', 'right'], default='both',
                        help='Stereo head eye(s) saved to disk; XR still receives the full stereo frame. '
                             'Use right to save color_0=physical right eye, color_1=left wrist, '
                             'color_2=right wrist, including when the server rotates the head image.')
    parser.add_argument('--task-dir', type=str, default='./utils/data/', help='path to save data')
    parser.add_argument('--task-name', type=str, default='ikea_task', help='task file name for recording')
    parser.add_argument('--task-goal', type=str, default='assemble the table.', help='task goal for recording')
    parser.add_argument('--task-desc', type=str, default='task description', help='task description')
    parser.add_argument('--task-steps', type=str, default='step1: do this; step2: do that;',
                        help="';'-separated subtask names; the Nth one is what number key [N] labels with")
    # extra recording channels (all on by default; opt out if a topic misbehaves)
    parser.add_argument('--no-lowstate-probe', action='store_true',
                        help='Skip whole-body dq/tau_est and the pelvis IMU; body qpos is still saved. '
                             'By default recording requires all 35 qpos/dq/tau_est slots.')
    parser.add_argument('--no-odom-probe', action='store_true',
                        help='Skip base odometry and twist from rt/odommodestate')
    parser.add_argument('--no-secondary-imu', action='store_true',
                        help='Skip the torso IMU (the head cameras\' own tilt)')
    parser.add_argument('--record-ee-pose', action='store_true',
                        help='Also write the full EE pose / pose_mat blocks. Off by default: they are a '
                             'pure function of the joints in the same row, so they belong in the '
                             'conversion step, where the frame convention can still be changed. '
                             'actions.*_ee.target_pose and ik_residual are written either way.')
    parser.add_argument('--no-ee-fk', action='store_true',
                        help='Skip online FK entirely, dropping ik_residual too. The pre-IK target is '
                             'still recorded -- it is the one EE quantity that cannot be recomputed.')
    parser.add_argument('--record-wireless-remote', action='store_true',
                        help='Also log the 40 raw bytes of the Unitree remote. Off by default because '
                             'XR-controller teleoperation leaves them at zero; the operator\'s real stick '
                             'input is recorded under actions.base.')
    parser.add_argument('--probe-wait', type=float, default=3.0,
                        help='Seconds to wait for each extra DDS topic. Recording requires lowstate '
                             'unless --no-lowstate-probe is explicitly set.')
    # heads-up notices inside the headset (pass-through only; see utils/xr_notify.py)
    parser.add_argument('--no-xr-notify', action='store_true',
                        help='Do not draw recording / camera-link notices in the headset. They are on '
                             'by default in --display-mode pass-through, where the operator cannot see '
                             'this terminal, and are ignored in the other display modes.')
    parser.add_argument('--xr-notify-elevation', type=float, default=22.0,
                        help='Degrees above the line of sight to place the notice banner. It has to '
                             'clear the working area without needing a head tilt to read; raise it if '
                             'the banner still crowds the view, lower it if you keep missing notices.')
    parser.add_argument('--xr-notify-width', type=float, default=34.0,
                        help='Angular width of the notice banner, in degrees')
    parser.add_argument('--xr-notify-distance', type=float, default=1.0,
                        help='Metres in front of the headset to place the notice banner')
    parser.add_argument('--camera-stale-s', type=float, default=1.5,
                        help='Seconds without a new head frame before the teleimager link is called '
                             'down. Raise it if a loaded network makes the notice flicker.')
    return parser


# Make shared, lightweight CLI modules available before the early --help path.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# TeleVuer's import-time CLI also defines --help. Answer our own help request
# before importing TeleVuer; normal startup is unchanged.
if __name__ == '__main__' and any(arg in ('-h', '--help') for arg in sys.argv[1:]):
    _build_arg_parser().parse_args()

# Import this repository's televuer rather than whichever editable install is on
# the path: the headset notice banner is implemented in the copy vendored here.
# The two were byte-identical before that banner, so this only makes the choice
# deterministic -- the resolved path is logged below to keep it checkable.
_televuer_src = os.path.join(parent_dir, 'televuer', 'src')
if os.path.isdir(_televuer_src) and _televuer_src not in sys.path:
    sys.path.insert(0, _televuer_src)
_early_args = None
if __name__ == "__main__":
    from teleop.utils.xr_server import ensure_xr_port_available
    _early_parser = _build_arg_parser()
    _early_args = _early_parser.parse_args()
    if not _early_args.motion:
        _early_parser.error("BOOST G1 teleop requires --motion; arm targets must use rt/arm_sdk")
    if _early_args.sim or _early_args.loco_client:
        _early_parser.error("BOOST G1 teleop is hardware-only; DreamWaQ owns locomotion and rt/lowcmd")
    try:
        ensure_xr_port_available()
    except (ValueError, RuntimeError) as error:
        _early_parser.error(str(error))

import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

# TeleVuer must be imported before unitree_sdk2py on this ARM64 environment.
from televuer import TeleVuerWrapper
from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds
from teleop.robot_control.robot_arm import G1_29_ArmController
from teleop.robot_control.waist_yaw import WAIST_RATE_RAD_S, WAIST_TURN_RIGHT_DEG, WAIST_TURN_RIGHT_RAD
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from teleop.utils.multi_host_image_client import (
    MultiHostImageClient,
    parse_camera_host_pairs,
    resolve_camera_hosts,
)
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.loop_rate import LoopRate
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from teleop.utils.dds_interface import pin_interface
from teleop.utils.robot_state_probe import RobotStateProbes
from teleop.utils import ee_kinematics
from teleop.utils.recording_contract import (
    recording_body_state,
    recording_camera_images,
    recording_camera_timestamp,
    recording_camera_layout,
)
from teleop.utils.xr_notify import (
    LEVEL_ERROR,
    LEVEL_GOOD,
    LEVEL_RECORDING,
    LEVEL_WARN,
    LINK_LOST,
    LINK_RESTORED,
    StreamHealthMonitor,
    XRNotifier,
)
from teleop.utils.teleop_keyboard import listen_teleop_keys

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
RECENTER_TOGGLE = False # Re-capture the head anchor used by --arm-reference-mode fixed
LABEL_ID       = 0      # Current subtask label; 0 means unlabeled
WAIST_TURNED   = False  # [w]: waist yaw turned WAIST_TURN_RIGHT_DEG to the robot's right (DreamWaQ applies it)
WAIST_AVAILABLE = False # only --motion has DreamWaQ to ask; set once the arm controller exists
KEYBOARD_STOP_EVENT = threading.Event()
LAST_SAVED_EPISODE = None
RECORDER = None

# Standby pose: where the arms wait before [r] and between episodes. All-zero is not
# arms-down -- the G1's zero already holds the hands forward at (0.25, +-0.15, 0.10) m
# in the pelvis frame -- it is just lower than is comfortable to start tracking from.
# Shoulder pitch alone lifts it: -15 deg puts the hands 7 cm higher and 4 cm further
# forward (measured with utils/ikea_urdf_fk.py against the G1_29 URDF). Watch the signs,
# they are not what the names suggest: shoulder pitch *negative* raises the hand, and
# elbow *negative* raises it further (positive elbow folds the forearm back down, since
# the zero-pose forearm already points forward). Index order is the 14-dof dual-arm
# vector -- 0..6 left (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw), 7..13 right.
PARK_Q = np.zeros(14)
PARK_Q[[0, 7]] = np.deg2rad(-15.0)   # shoulder pitch, both arms


def parse_task_steps(text):
    """--task-steps text -> {1: name, 2: name, ...}, the map the number keys select.

    Absolute ids rather than a "next step" advance, because an assembly sequence
    repeats: the same leg-screwing skill happens four times, and pressing the same
    digit again is the whole story. A monotonic advance would need one id per
    repetition, which turns one skill with four demonstrations into four classes
    with one each.

    The string is already written to data.json's text.steps, so the map that was
    labelled with travels with the episode instead of living in shell history.
    """
    steps = [step.strip() for step in (text or "").split(";")]
    return {i: step for i, step in enumerate((step for step in steps if step), start=1)}


LABEL_NAMES = {}


def on_press(key):
    global STOP, START, RECORD_TOGGLE, RECENTER_TOGGLE, LABEL_ID, WAIST_TURNED
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
    elif key == 'c':
        RECENTER_TOGGLE = True
    elif key == 'w':
        if WAIST_AVAILABLE:
            WAIST_TURNED = not WAIST_TURNED
        else:
            logger_mp.warning("[waist] [w] needs --motion: DreamWaQ owns the waist and applies the turn")
    elif len(key) == 1 and key.isdigit():
        # Labelling is a second operator's job at the monitor, so an unknown digit
        # is recorded rather than ignored: a typo that silently leaves the previous
        # label in place is invisible later, whereas an unnamed id is not.
        LABEL_ID = int(key)
        if LABEL_ID == 0:
            logger_mp.info("[label] 0 -> (unlabeled)")
        elif LABEL_ID in LABEL_NAMES:
            logger_mp.info(f"[label] {LABEL_ID} -> {LABEL_NAMES[LABEL_ID]}")
        else:
            logger_mp.warning(f"[label] {LABEL_ID} has no name in --task-steps; recorded anyway")
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")


def on_short_q():
    """A short q marks the last saved episode as failure; it does not stop."""
    if LAST_SAVED_EPISODE is None or RECORDER is None:
        logger_mp.info("No saved episode to relabel; hold [q] for 3 seconds to stop.")
        return
    try:
        RECORDER.set_episode_success(LAST_SAVED_EPISODE, False)
        logger_mp.info(f"Marked the last saved episode as failure: {LAST_SAVED_EPISODE}")
    except Exception as error:
        logger_mp.error(f"Could not mark the last saved episode as failure: {error}")


def on_long_q():
    logger_mp.info("[q] held for 3 seconds: stopping teleoperation.")
    on_press('q')


def run_keyboard_listener():
    try:
        listen_teleop_keys(on_press, on_short_q, on_long_q, KEYBOARD_STOP_EVENT)
    except Exception as error:
        logger_mp.error(f"Terminal key listener failed: {error}")


def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY, LABEL_ID
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
        "LABEL_ID": LABEL_ID,
        "LABEL": LABEL_NAMES.get(LABEL_ID, ""),
        "WAIST_TURNED": WAIST_TURNED,
    }


def go_home_and_release(arm_ctrl, timeout=15.0, tolerance=0.05):
    """Drive both arms to the home pose on the way out, then hand control back.

    ctrl_dual_arm_go_home() alone is not enough to end a session at home: it gives
    up after a fixed 100 * 0.05 s = 5 s, and in --motion mode it only ramps the arm
    weight (kNotUsedJoint0) back down to the locomotion controller *inside* the
    reached-home branch. So an arm that is still travelling when those 5 s run out
    ends the session wherever it happened to be, with the process exiting straight
    afterwards -- the control thread is a daemon, so nothing keeps commanding it.

    Waiting for home here first means the call below finds the arms already inside
    tolerance on its first check and only performs the handover, which then starts
    from a known pose.
    """
    zeros = np.zeros(14)
    arm_ctrl.ctrl_dual_arm(zeros, zeros)
    if getattr(arm_ctrl, "waist_yaw", None) is not None:
        arm_ctrl.set_waist_yaw_target(0.0)
    deadline = time.time() + timeout
    reached = False
    while time.time() < deadline:
        if np.all(np.abs(arm_ctrl.get_current_dual_arm_q()) < tolerance):
            reached = True
            break
        time.sleep(0.05)
    if not reached:
        logger_mp.warning(f"Arms did not reach the home position within {timeout}s; "
                          "handing control back from wherever they are.")
    arm_ctrl.ctrl_dual_arm_go_home()


def _f(values):
    """DDS arrays and numpy rows -> plain float lists, so json.dumps never chokes."""
    if values is None:
        return None
    return [float(v) for v in values]


def _sub(values, start, end):
    return None if values is None else _f(values[start:end])


def _pose_fields(T, enabled):
    """Euler vector plus the lossless 4x4, only when explicitly asked for.

    Off by default. These are FK of the joints in the same row, so writing them
    here would freeze one frame convention into the recording for no information
    gain; the conversion step can produce this and a waist-included variant from
    body.qpos whenever it wants.
    """
    if T is None or not enabled:
        return {}
    return {"pose": ee_kinematics.mat_to_pose_vec(T),
            "pose_mat": ee_kinematics.mat_to_flat(T)}


if __name__ == '__main__':
    args = _early_args
    logger_mp.debug(f"args: {args}")

    LABEL_NAMES = parse_task_steps(args.task_steps)
    if args.record:
        logger_mp.info("Subtask labels (number keys, sticky until the next digit; [0] = unlabeled):")
        for label_id, name in sorted(LABEL_NAMES.items()):
            logger_mp.info(f"    [{label_id}] {name}")

    img_client = None
    rec_img_client = None
    xr_notifier = None      # referenced by the cleanup block below
    probes = None
    arm_ctrl = None

    try:
        # setup dds communication domains id. The interface goes through CYCLONEDDS_URI
        # rather than the SDK's networkInterface argument, whose inline XML config
        # aborts the process inside an activated conda env (see utils/dds_interface.py).
        pin_interface(args.network_interface, logger_mp)
        ChannelFactoryInitialize(1 if args.sim else 0)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press, get_state=get_state)
            ipc_server.start()
        else:
            listen_keyboard_thread = threading.Thread(target=run_keyboard_listener, daemon=True)
            listen_keyboard_thread.start()

        # image client
        camera_hosts = resolve_camera_hosts(args.img_server_ip, parse_camera_host_pairs(args.camera_host_for))
        logger_mp.info("Camera hosts: " + ", ".join(f"{name}={host}" for name, host in camera_hosts.items()))
        img_client = MultiHostImageClient(camera_hosts, request_bgr=True)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")
        if args.record:
            record_camera_layout = recording_camera_layout(camera_config, args.record_head_eye)
            if args.record_head_eye != 'both':
                missing_wrists = [name for name in ('left_wrist_camera', 'right_wrist_camera')
                                  if not any(source['camera'] == name
                                             for source in record_camera_layout.values())]
                if missing_wrists:
                    raise RuntimeError("Single-eye recording requires both wrist camera streams: "
                                       + ', '.join(missing_wrists))
        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])

        # Prefer the timestamped TIM2 streams when the server publishes them: they
        # carry the capture time each frame needs to be aligned with its action,
        # and they let the original JPEG reach disk without a re-encode.
        head_needs_tim2 = xr_need_local_img and not camera_config['head_camera']['enable_zmq']
        if any(isinstance(cam, dict) and cam.get('enable_recording_zmq', False)
               for cam in camera_config.values()) and (args.record or head_needs_tim2):
            from teleop.utils.recording_image_client import RecordingImageClient

            wanted = None if args.record else ('head_camera',)
            if args.record:
                # One transport for the whole episode. With the cameras split over
                # two servers, one of them may run without --rs; its camera would
                # otherwise just be missing from every frame on disk.
                no_tim2 = [f"{name} on {camera_hosts[name]}" for name, cam in camera_config.items()
                           if (cam.get('enable_zmq', False) or cam.get('enable_webrtc', False))
                           and not cam.get('enable_recording_zmq', False)]
                if no_tim2:
                    raise RuntimeError("Recording uses the TIM2 streams, but these cameras have none "
                                       "(enable_recording_zmq must be true in their server YAML): "
                                       + ', '.join(no_tim2))
            rec_img_client = RecordingImageClient(camera_hosts, camera_config, cameras=wanted,
                                                 clock_sync=args.camera_clock_sync)
            if not rec_img_client.wait_until_ready(timeout=10.0):
                if args.record:
                    raise RuntimeError("Not all recording cameras delivered a frame; check their "
                                       "server YAML, connections and --camera-host-for addresses.")
                logger_mp.warning("Some recording streams have not delivered a frame yet; "
                                  "check the server recording streams.")

        if rec_img_client is not None:
            get_head_frame  = rec_img_client.get_head_frame
            get_left_frame  = rec_img_client.get_left_wrist_frame
            get_right_frame = rec_img_client.get_right_wrist_frame
            head_img_enabled  = rec_img_client.has('head_camera')
            left_img_enabled  = rec_img_client.has('left_wrist_camera')
            right_img_enabled = rec_img_client.has('right_wrist_camera')
        else:
            get_head_frame  = img_client.get_head_frame
            get_left_frame  = img_client.get_left_wrist_frame
            get_right_frame = img_client.get_right_wrist_frame
            head_img_enabled  = camera_config['head_camera']['enable_zmq']
            left_img_enabled  = camera_config['left_wrist_camera']['enable_zmq']
            right_img_enabled = camera_config['right_wrist_camera']['enable_zmq']

        # Headset notices. Only pass-through hides the terminal from the operator
        # *and* leaves the scene free to draw on, so that is the mode they are
        # built for; the banner is skipped elsewhere rather than fighting the
        # camera image for the same plane.
        camera_health = None
        tv_notify_kwargs = {}
        if not args.no_xr_notify and args.display_mode == 'pass-through':
            import inspect as _inspect

            import televuer as _televuer

            # An older televuer elsewhere on the path has no notifier parameter,
            # and passing one would abort a run that is otherwise fine. Check
            # first and degrade to a warning instead.
            if 'notifier' in _inspect.signature(TeleVuerWrapper.__init__).parameters:
                xr_notifier = XRNotifier(elevation_deg=args.xr_notify_elevation,
                                         width_deg=args.xr_notify_width,
                                         distance_to_camera=args.xr_notify_distance)
                camera_health = StreamHealthMonitor(stale_after_s=args.camera_stale_s)
                tv_notify_kwargs['notifier'] = xr_notifier
                logger_mp.info(f"XR notices enabled at {args.xr_notify_elevation:.0f} deg above "
                               f"the line of sight (televuer from {_televuer.__file__})")
            else:
                logger_mp.warning(
                    f"XR notices unavailable: televuer at {_televuer.__file__} predates the "
                    "notice banner. Teleoperation continues without headset notices.")

        # televuer_wrapper: XR pose in, head camera image out.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand",
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     display_mode=args.display_mode,
                                     zmq=head_img_enabled and xr_need_local_img,
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=(args.head_webrtc_url or
                                                 f"https://{camera_hosts['head_camera']}:{camera_config['head_camera']['webrtc_port']}/offer"),
                                     arm_reference_mode=('head_position' if args.arm_reference_mode == 'fixed'
                                                         else args.arm_reference_mode),
                                     track_motion_timestamp=True,
                                     **tv_notify_kwargs
                                     )

        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        loco_wrapper = None
        if args.motion:
            if args.input_mode != "controller":
                logger_mp.warning("--motion without --input-mode controller: no locomotion command can be "
                                  "issued, so actions.base.cmd_vel will stay zero for the whole session.")
            elif args.loco_client:
                loco_wrapper = LocoClientWrapper()
            else:
                logger_mp.info("--motion without --loco-client: publishing arm targets on rt/arm_sdk, but "
                                "not driving base locomotion via LocoClient RPC (assuming an external "
                                "low-level controller, e.g. HOMIE, owns legs/waist instead).")
        elif args.skip_motion_switch:
            logger_mp.info("Skipping Enter_Debug_Mode (--skip-motion-switch): assuming an external "
                            "low-level controller already owns rt/lowcmd.")
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        # arm
        arm_ik = G1_29_ArmIK()
        arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)

        # The controller's q_target starts at zeros and its publish thread is already
        # running, so the arms begin ramping to "straight down" the moment it is
        # constructed; overriding the target here redirects that first ramp to the
        # standby pose instead of adding a second move.
        arm_ctrl.ctrl_dual_arm(PARK_Q, np.zeros(14))
        WAIST_AVAILABLE = getattr(arm_ctrl, "waist_yaw", None) is not None

        # EE forward kinematics through the IK's own model. Its only job by default
        # is ik_residual -- a live, 4-number check that the arm is going where the
        # operator points. The poses themselves stay out of the recording unless
        # --record-ee-pose asks for them, because the conversion step can derive
        # them (in this frame or a waist-included one) from body.qpos.
        ee_fk = None
        if not args.no_ee_fk:
            try:
                ee_fk = ee_kinematics.ArmEEKinematics(arm_ik)
                logger_mp.info(f"[ee_fk] enabled, frame: {ee_kinematics.EE_FRAME_CONVENTION['reference_link']} / "
                               f"{ee_kinematics.EE_FRAME_CONVENTION['waist_joints']} / "
                               f"+{ee_kinematics.EE_FRAME_CONVENTION['ee_offset_xyz_m'][0]} m on wrist_yaw x"
                               f"{'; writing pose blocks' if args.record_ee_pose else '; ik_residual only'}")
            except Exception as error:
                logger_mp.warning(f"[ee_fk] disabled: {error}")

        # Extra read-only DDS probes. Started after the controllers so a slow topic
        # cannot delay the robot coming up.
        probes = RobotStateProbes(enable_lowstate=not args.no_lowstate_probe,
                                  enable_odom=not args.no_odom_probe,
                                  enable_secondary_imu=not args.no_secondary_imu,
                                  enable_gripper=False,
                                  num_joints=BODY_MOTOR_SLOTS,
                                  wait_timeout=args.probe_wait,
                                  with_wireless_remote=args.record_wireless_remote)
        logger_mp.info(f"[probes] {probes.status()}")
        if args.record and not args.no_lowstate_probe:
            # A failed probe is permanently disabled. Refuse an empty dataset at
            # startup rather than entering an episode that can never save rows.
            recording_body_state(arm_ctrl.get_current_motor_q().tolist(),
                                 probes.snapshot()['lowstate'], motor_slots=BODY_MOTOR_SLOTS)

        # affinity mode
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0, 1, 2, 3])
            try:
                p.nice(-20)
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5, 6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            recorder = EpisodeWriter(task_dir=os.path.join(args.task_dir, args.task_name),
                                     task_goal=args.task_goal,
                                     task_desc=args.task_desc,
                                     task_steps=args.task_steps,
                                     frequency=args.frequency,
                                     rerun_log=not args.headless,
                                     compact_json=True, collect_stats=True)
            RECORDER = recorder
            # Stamp how the numbers in this episode were produced. A dataset that
            # does not carry its own frame convention is a dataset whose EE poses
            # have to be reverse engineered later.
            recorder.info["recorder"] = "boost_arm_only.py"
            recorder.info["camera_config"] = camera_config
            recorder.info["camera_hosts"] = camera_hosts
            recorder.info["camera_image_keys"] = record_camera_layout
            recorder.info["record_head_eye"] = args.record_head_eye
            recorder.info["camera_clock_calibrations"] = (
                {host: clock.metadata() for host, clock in rec_img_client.clock_calibrations.items()}
                if rec_img_client is not None else {}
            )
            recorder.info["labels"] = {str(k): v for k, v in LABEL_NAMES.items()}
            recorder.info["label_note"] = (
                "annotation.label_id per frame, set live with the number keys; 0 is unlabeled. "
                "Sticky: it holds until the next digit, so a repeated subtask is the same id again.")
            recorder.info["arm"] = args.arm
            recorder.info["ee"] = "none; finger control disabled"
            recorder.info["joint_names"] = {
                "left_arm": list(G1_JOINT_NAMES[15:22]),
                "right_arm": list(G1_JOINT_NAMES[22:29]),
                "left_ee": [],
                "right_ee": [],
                # `actions.body.qpos` is the historical cmd_vel vector, not joints.
                "body": [],
            }
            recorder.info["g1_body_joint_names"] = list(G1_JOINT_NAMES)
            recorder.info["input_mode"] = args.input_mode
            recorder.info["motion_mode"] = bool(args.motion)
            recorder.info["loco_velocity_scale"] = LOCO_VELOCITY_SCALE
            recorder.info["body_motor_slots"] = BODY_MOTOR_SLOTS
            recorder.info["body_motor_note"] = "indices 0..28 are the G1_29 joints; 29..34 are unused padding"
            recorder.info["waist_command"] = {
                "available": WAIST_AVAILABLE,
                "key": "w",
                "turn_rad": WAIST_TURN_RIGHT_RAD,
                "rate_rad_s": WAIST_RATE_RAD_S,
                "note": ("actions.waist.qpos is the waist [yaw, roll, pitch] requested from DreamWaQ "
                         "on rt/arm_sdk (slot 12, enabled by slot 30); DreamWaQ applies it and hides it "
                         "from its own policy. The arm IK stays waist-zero, so the arms turn with the "
                         "torso. states.body.qpos[12] is the measured waist yaw."),
            }
            recorder.info["whole_body_state_required"] = not args.no_lowstate_probe
            # The convention behind actions.*_ee.target_pose, ik_residual, and the
            # optional pose blocks -- and the frame the conversion step should
            # reproduce when it computes EE poses from body.qpos.
            recorder.info["ee_frame"] = ee_kinematics.frame_convention(arm_ik) if ee_fk else None
            recorder.info["ee_pose_recorded"] = bool(ee_fk and args.record_ee_pose)
            recorder.info["dds_topics"] = {
                "lowstate": "rt/lowstate",
                "odom": "rt/odommodestate",
                "secondary_imu": "rt/secondary_imu",
                "arm_command": "rt/arm_sdk",
                "note": "full-rate topics only; the rt/lf/* mirrors publish at 20 Hz",
            }
            recorder.info["base_pose_note"] = (
                "states.base.odom_pos / odom_quat are a session-local, drifting odom frame. "
                "Useful for segmentation and evaluation; not intended as a policy input. "
                "Use lin_vel / ang_vel / projected_gravity for that.")

        def park_arms(timeout=5.0, tolerance=0.05):
            '''Send both arms to the standby pose and wait until they arrive.

            arm_ctrl.ctrl_dual_arm_go_home() is deliberately not used here: in
            --motion mode it ramps kNotUsedJoint0 down to 0, handing the arms back
            to the locomotion controller, and nothing restores that weight.
            '''
            global WAIST_TURNED
            arm_ctrl.ctrl_dual_arm(PARK_Q, np.zeros(PARK_Q.shape[0]))
            # Every episode starts at the default waist, like an unlabeled head.
            WAIST_TURNED = False
            apply_waist_turn()
            deadline = time.time() + timeout
            while time.time() < deadline:
                if np.all(np.abs(arm_ctrl.get_current_dual_arm_q() - PARK_Q) < tolerance):
                    logger_mp.info("Both arms have reached the standby position.")
                    return True
                time.sleep(0.05)
            logger_mp.warning(f"Arms did not reach the standby position within {timeout}s; continuing.")
            return False

        logger_mp.info("----------------------------------------------------------------")
        xr_host = os.environ.get("BOOST_G1_XR_HOST", "<G1_TELEOP_HOST_IP>")
        logger_mp.info("XR control page (including pass-through): "
                       f"https://{xr_host}:8012/?ws=wss://{xr_host}:8012 ; "
                       "enter the XR session there before pressing [r].")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
            logger_mp.info("🟡  Saving parks the robot at home; press [r] to resume tracking.")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        if WAIST_AVAILABLE:
            logger_mp.info(f"🟣  Press [w] to turn the waist {WAIST_TURN_RIGHT_DEG:.0f} deg right, [w] again to return "
                           "(DreamWaQ applies it; the arms turn with the torso; saving resets it).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")

        # ---- headset notices ------------------------------------------------
        # None of the lines above reach an operator wearing the headset in
        # pass-through, so the same state is mirrored into the XR scene from
        # here on. All of it is a no-op when notices are off.
        record_started_at = None
        pending_save = None      # (episode_id, frames) handed to the writer, not yet on disk
        last_status_push = 0.0

        def notify_camera_link(frame):
            """Report head-stream link edges on the alert channel.

            Using the alert channel rather than the status line is deliberate: a
            dead camera then covers the recording indicator instead of the two
            taking turns, and it survives until the link actually comes back.
            """
            if camera_health is None:
                return
            edge = camera_health.update(frame)
            if edge == LINK_LOST:
                logger_mp.error(f"teleimager head stream is down "
                                f"({camera_health.down_for_s():.1f}s without a frame)")
                xr_notifier.alert("CAMERA LINK LOST - check teleimager server",
                                  LEVEL_ERROR, ttl_s=None)
            elif edge == LINK_RESTORED:
                logger_mp.info("teleimager head stream recovered")
                xr_notifier.alert("CAMERA LINK OK", LEVEL_GOOD, ttl_s=3.0)

        def notify_save_completed():
            """Announce an episode once the writer has actually closed the file.

            save_episode() only raises a flag, so the operator's [s] is not proof
            anything landed; is_ready() going back up is. Reported from both the
            parked wait loop and the main loop because either can be the one that
            is running when the writer finishes.
            """
            global pending_save
            if pending_save is None or not recorder.is_ready():
                return False
            saved_id, saved_frames = pending_save
            pending_save = None
            logger_mp.info(f"==> episode_{saved_id:04d} closed with {saved_frames} frames")
            if xr_notifier is not None:
                xr_notifier.alert(f"SAVED episode_{saved_id:04d} - {saved_frames} frames",
                                  LEVEL_GOOD, ttl_s=6.0)
            return True

        def notify_state(force=False):
            """Mirror recorder / tracking state onto the sticky notice line.

            Throttled because the frame counter changes every control cycle and
            each change costs the Vuer process one banner encode. Transitions
            are announced on the alert channel anyway, so a fraction of a second
            of lag on the status line is not visible.
            """
            global last_status_push
            if xr_notifier is None:
                return
            now = time.time()
            if not force and now - last_status_push < 0.4:
                return
            last_status_push = now
            # The operator in the headset cannot see the terminal, and a turned
            # waist moves both arms with the torso.
            waist = f" | WAIST R{WAIST_TURN_RIGHT_DEG:.0f}" if WAIST_TURNED else ""
            if not args.record:
                xr_notifier.status(("TELEOP RUNNING" if START else "PAUSED - press [r] to track") + waist)
            elif RECORD_RUNNING:
                elapsed = 0.0 if record_started_at is None else now - record_started_at
                label = LABEL_NAMES.get(LABEL_ID, "unlabeled" if LABEL_ID == 0 else f"label {LABEL_ID}")
                xr_notifier.status(
                    f"REC episode_{recorder.episode_id:04d} | {recorder.item_id + 1} frames"
                    f" | {elapsed:.0f} s | {label}{waist}", LEVEL_RECORDING)
            elif pending_save is not None:
                xr_notifier.status(f"SAVING episode_{pending_save[0]:04d} ...", LEVEL_WARN)
            elif START:
                xr_notifier.status("STANDBY - press [s] to start recording" + waist)
            else:
                xr_notifier.status("PARKED - press [r] to track, [s] to record")

        applied_waist_turned = False

        def apply_waist_turn():
            """Hand a [w] toggle to the arm controller, which ramps the request."""
            global applied_waist_turned
            if not WAIST_AVAILABLE or WAIST_TURNED == applied_waist_turned:
                return
            arm_ctrl.set_waist_yaw_target(WAIST_TURN_RIGHT_RAD if WAIST_TURNED else 0.0)
            applied_waist_turned = WAIST_TURNED
            text = f"WAIST -> {WAIST_TURN_RIGHT_DEG:.0f} deg RIGHT" if WAIST_TURNED else "WAIST -> default"
            logger_mp.info(f"[waist] {text} (ramped at {WAIST_RATE_RAD_S} rad/s, applied by DreamWaQ)")
            if xr_notifier is not None:
                xr_notifier.alert(text, LEVEL_WARN, ttl_s=3.0)
            notify_state(force=True)

        READY = True
        notify_state(force=True)
        while not STOP:
            apply_waist_turn()
            if START:
                break
            time.sleep(0.033)
            if head_img_enabled and (xr_need_local_img or camera_health is not None):
                head_img = get_head_frame()
                if xr_need_local_img and head_img is not None and head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)
                notify_camera_link(head_img)

        if not STOP:
            logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
            arm_ctrl.speed_gradual_max()

        head_img = None
        left_wrist_img = None
        right_wrist_img = None
        last_recording_gap_notice = float('-inf')
        last_xr_input_notice = float('-inf')
        xr_input_was_ready = False
        head_anchor_pos = None

        # main loop. robot start to follow VR user's motion
        loop_rate = LoopRate(args.frequency)
        while not STOP:
            # get image
            if head_img_enabled:
                if args.record or xr_need_local_img or camera_health is not None:
                    head_img = get_head_frame()
                if xr_need_local_img and head_img is not None and head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)
                notify_camera_link(head_img)
            if left_img_enabled and args.record:
                left_wrist_img = get_left_frame()
            if right_img_enabled and args.record:
                right_wrist_img = get_right_frame()
            notify_state()
            apply_waist_turn()

            # Pass-through still needs live XR pose events. The wrapper supplies
            # fallback poses before the first packet; solving IK for those poses
            # makes the robot move even though the headset controls nothing.
            tele_data = tv_wrapper.get_tele_data()
            motion_stamp = tele_data.motion_data_monotonic_ns
            now_xr = time.monotonic_ns()
            xr_input_ready = (tele_data.motion_data_ready and motion_stamp > 0
                              and 0 <= now_xr - motion_stamp <= XR_INPUT_TIMEOUT_S * 1e9)
            if not xr_input_ready:
                head_anchor_pos = None
                # Saving an active episode must remain possible after XR pose
                # loss. Starting a new episode still requires fresh poses.
                if args.record and RECORD_TOGGLE:
                    RECORD_TOGGLE = False
                    if RECORD_RUNNING:
                        RECORD_RUNNING = False
                        pending_save = (recorder.episode_id, recorder.item_id + 1)
                        record_started_at = None
                        LAST_SAVED_EPISODE = os.path.abspath(recorder.json_path)
                        recorder.save_episode(final_metadata={
                            "success": True, "label": "success", "save_requested": True,
                            "xr_input_missing": True,
                        })
                        START = False
                        READY = recorder.is_ready()
                        logger_mp.info("XR input is missing; saving the active episode and parking the arms.")
                        park_arms()
                        notify_state(force=True)
                    else:
                        logger_mp.warning("Cannot start recording until fresh XR poses are available.")
                if args.record:
                    READY = recorder.is_ready()
                    notify_save_completed()
                if time.monotonic() - last_xr_input_notice >= 2.0:
                    logger_mp.warning("XR INPUT MISSING: holding the last arm target; IK/recording paused. "
                                      "Open Thor's :8012 XR page, enter the XR session and use "
                                      f"{args.input_mode} tracking (display={args.display_mode}).")
                    if xr_notifier is not None:
                        xr_notifier.status("XR INPUT MISSING - enter XR session", LEVEL_ERROR)
                    last_xr_input_notice = time.monotonic()
                xr_input_was_ready = False
                loop_rate.sleep()
                continue
            if not xr_input_was_ready:
                logger_mp.info(f"XR INPUT READY: live {args.input_mode} poses received; arm tracking enabled.")
                xr_input_was_ready = True

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                        record_started_at = time.time()
                        # Not carried over from the previous episode: a stale label
                        # leaking into a new recording is silent, an unlabeled head
                        # is visible in the converter's report.
                        LABEL_ID = 0
                        if xr_notifier is not None:
                            xr_notifier.alert(f"RECORDING episode_{recorder.episode_id:04d}",
                                              LEVEL_RECORDING, ttl_s=3.0)
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                        if xr_notifier is not None:
                            # The writer refuses a new episode while the previous
                            # one is still draining, which is the one failure the
                            # operator can act on by simply waiting.
                            xr_notifier.alert("RECORD START FAILED - previous episode still saving",
                                              LEVEL_ERROR, ttl_s=6.0)
                else:
                    RECORD_RUNNING = False
                    # save_episode() only raises a flag; the writer thread drains
                    # the queue and flips is_ready() when the file is closed. Hold
                    # the counts so the notice can report what actually landed.
                    pending_save = (recorder.episode_id, recorder.item_id + 1)
                    record_started_at = None
                    LAST_SAVED_EPISODE = os.path.abspath(recorder.json_path)
                    recorder.save_episode(final_metadata={
                        "success": True, "label": "success", "save_requested": True,
                    })
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

                    START = False
                    READY = recorder.is_ready()
                    notify_state(force=True)
                    logger_mp.info("-------------------⏸️ episode saved ⏸️---------------------")
                    park_arms()
                    logger_mp.info("🟢  Robot parked at home. Press [r] to resume tracking, [q] to exit.")
                    while not START and not STOP:
                        time.sleep(0.033)
                        if head_img_enabled and (xr_need_local_img or camera_health is not None):
                            head_img = get_head_frame()
                            if xr_need_local_img and head_img is not None and head_img.bgr is not None:
                                tv_wrapper.render_to_xr(head_img.bgr)
                            notify_camera_link(head_img)
                        if notify_save_completed():
                            READY = True
                        notify_state()
                        apply_waist_turn()
                    if STOP:
                        break
                    logger_mp.info("---------------------🚀resume Tracking🚀------------------------")
                    arm_ctrl.speed_gradual_max()
                    head_anchor_pos = None
                    loop_rate.reset()
                    continue

            # high level control. The command is built once and reused for the
            # recording, so what lands in the file is what was actually sent
            # rather than a second evaluation of the same expression.
            cmd_vel = [0.0, 0.0, 0.0]
            cmd_vel_sent = False
            left_stick = [0.0, 0.0]
            right_stick = [0.0, 0.0]
            if args.input_mode == "controller":
                left_stick = _f(tele_data.left_ctrl_thumbstickValue)
                right_stick = _f(tele_data.right_ctrl_thumbstickValue)
                if args.motion:
                    if tele_data.right_ctrl_aButton:
                        START = False
                        STOP = True
                    cmd_vel = [-left_stick[1] * LOCO_VELOCITY_SCALE,
                               -left_stick[0] * LOCO_VELOCITY_SCALE,
                               -right_stick[0] * LOCO_VELOCITY_SCALE]
                    if loco_wrapper is not None:
                        # soft emergency stop
                        if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                            loco_wrapper.Damp()
                        loco_wrapper.Move(cmd_vel[0], cmd_vel[1], cmd_vel[2])
                        cmd_vel_sent = True

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # Anchor the wrist targets to a fixed head pose. televuer expresses each
            # target as (hand_world - head_world), so any head motion drags the arms
            # along; substituting a frozen anchor is a pure translation.
            left_wrist_target  = tele_data.left_wrist_pose
            right_wrist_target = tele_data.right_wrist_pose
            if args.arm_reference_mode == 'fixed':
                head_pos = tele_data.head_pose[:3, 3]
                if head_anchor_pos is None or RECENTER_TOGGLE:
                    RECENTER_TOGGLE = False
                    head_anchor_pos = head_pos.copy()
                    logger_mp.info(f"[arm reference] head anchor captured at {head_anchor_pos.round(3).tolist()}")
                head_offset = head_pos - head_anchor_pos
                left_wrist_target  = left_wrist_target.copy()
                right_wrist_target = right_wrist_target.copy()
                left_wrist_target[:3, 3]  += head_offset
                right_wrist_target[:3, 3] += head_offset

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            sol_q, sol_tauff = arm_ik.solve_ik(left_wrist_target, right_wrist_target,
                                               current_lr_arm_q, current_lr_arm_dq)
            time_ik_end = time.time()
            action_wall_time_us = int(time_ik_end * 1e6)
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            # record data
            if args.record:
                READY = recorder.is_ready()
                # Reached when [r] resumed tracking before the writer had
                # finished draining the previous episode.
                notify_save_completed()

                if RECORD_RUNNING:
                    # Everything below is only needed for a frame that is actually
                    # being written, so it stays out of the idle path.
                    probe_data = probes.snapshot(action_wall_time_us)
                    low = probe_data["lowstate"]
                    odom = probe_data["odom"]
                    torso = probe_data["secondary_imu"]

                    # Whole-body motor state. qpos keeps coming from the arm
                    # controller so its meaning and length are unchanged; qvel and
                    # torque come from the probe over the same 35 slots, so index i
                    # is the same joint in all three.
                    current_body_state = arm_ctrl.get_current_motor_q().tolist()
                    try:
                        body_state = recording_body_state(
                            current_body_state, low, require_full=not args.no_lowstate_probe,
                            motor_slots=BODY_MOTOR_SLOTS)
                        colors = recording_camera_images(camera_config, {
                            'head_camera': head_img,
                            'left_wrist_camera': left_wrist_img,
                            'right_wrist_camera': right_wrist_img,
                        }, args.record_head_eye)
                    except ValueError as error:
                        now = time.monotonic()
                        if now - last_recording_gap_notice >= 2.0:
                            logger_mp.warning(f"Skipping incomplete recording row: {error}")
                            last_recording_gap_notice = now
                        loop_rate.sleep()
                        continue
                    depths = {}

                    # arm state and action
                    left_arm_state  = current_lr_arm_q[:7]
                    right_arm_state = current_lr_arm_q[-7:]
                    left_arm_vel    = current_lr_arm_dq[:7]
                    right_arm_vel   = current_lr_arm_dq[-7:]
                    left_arm_action = sol_q[:7]
                    right_arm_action = sol_q[-7:]
                    left_arm_tauff  = sol_tauff[:7]
                    right_arm_tauff = sol_tauff[-7:]

                    # EE poses. The state pose is FK of the very joints recorded in
                    # this same row, which is what makes an EE/joint desync
                    # impossible here.
                    ee_state_pose = {"left": None, "right": None}
                    ee_action_pose = {"left": None, "right": None}
                    ik_residual = None
                    if ee_fk is not None:
                        try:
                            l_state_T, r_state_T = ee_fk.poses(current_lr_arm_q)
                            l_action_T, r_action_T = ee_fk.poses(sol_q)
                            ee_state_pose = {"left": l_state_T, "right": r_state_T}
                            ee_action_pose = {"left": l_action_T, "right": r_action_T}
                            l_pos_mm, l_rot_deg = ee_kinematics.pose_error(left_wrist_target, l_action_T)
                            r_pos_mm, r_rot_deg = ee_kinematics.pose_error(right_wrist_target, r_action_T)
                            ik_residual = {
                                "left_pos_mm": round(l_pos_mm, 4),
                                "left_rot_deg": round(l_rot_deg, 4),
                                "right_pos_mm": round(r_pos_mm, 4),
                                "right_rot_deg": round(r_rot_deg, 4),
                            }
                        except Exception as error:
                            logger_mp.warning(f"[ee_fk] skipped this frame: {error}")

                    # Arm torque slots inside the whole-body tau_est vector.
                    low_tau = low["tau_est"] if low else None

                    states = {
                        "left_arm": {
                            "qpos":   left_arm_state.tolist(),
                            "qvel":   left_arm_vel.tolist(),
                            # tau_est for the arm slots; previously always empty.
                            "torque": _sub(low_tau, 15, 22) or [],
                        },
                        "right_arm": {
                            "qpos":   right_arm_state.tolist(),
                            "qvel":   right_arm_vel.tolist(),
                            "torque": _sub(low_tau, 22, 29) or [],
                        },
                        "left_ee": {
                            "qpos":   [],
                            "qvel":   [],
                            "torque": [],
                            **_pose_fields(ee_state_pose["left"], args.record_ee_pose),
                        },
                        "right_ee": {
                            "qpos":   [],
                            "qvel":   [],
                            "torque": [],
                            **_pose_fields(ee_state_pose["right"], args.record_ee_pose),
                        },
                        "body": body_state,
                        # Everything a lower-body policy conditions on. projected_gravity
                        # is the yaw-invariant, sign-stable form of the pelvis attitude;
                        # the raw quaternion is kept beside it for traceability only.
                        "base": {
                            "lin_vel":           (odom or {}).get("velocity"),
                            "ang_vel":           (low or {}).get("imu_gyro"),
                            "yaw_speed":         (odom or {}).get("yaw_speed"),
                            "projected_gravity": (low or {}).get("projected_gravity"),
                            "rpy":               (low or {}).get("imu_rpy"),
                            "quat":              (low or {}).get("imu_quat"),
                            "accel":             (low or {}).get("imu_accel"),
                            # Session-local odom. Analysis only -- see info.base_pose_note.
                            "odom_pos":          (odom or {}).get("position"),
                            "odom_rpy":          (odom or {}).get("imu_rpy"),
                            "odom_quat":         (odom or {}).get("imu_quat"),
                        },
                        # Torso IMU: the head cameras are mounted on torso_link, so this
                        # is their own tilt. Its yaw integrates separately from the
                        # pelvis IMU's, so never difference the two yaws to get the
                        # waist angle -- body.qpos[12:15] is the waist.
                        "torso": {
                            "projected_gravity": (torso or {}).get("projected_gravity"),
                            "rpy":               (torso or {}).get("imu_rpy"),
                            "quat":              (torso or {}).get("imu_quat"),
                            "ang_vel":           (torso or {}).get("imu_gyro"),
                        },
                        # Session flags. motion_data_ready is the one that matters for
                        # filtering: when the XR stops delivering poses the target
                        # freezes while the arm keeps its last command, and those frames
                        # should not become training labels.
                        "teleop": {
                            "motion_data_ready": bool(tele_data.motion_data_ready),
                            "tracking_started":  bool(START),
                            "waist_turned":      bool(WAIST_TURNED),
                        },
                    }
                    # The [w] request as sent (ramped) and its toggle target, in the
                    # waist's motor order [yaw, roll, pitch]; DreamWaQ holds roll and
                    # pitch at default. Zeros without --motion, so every episode has it.
                    waist_sent = arm_ctrl.get_waist_yaw() if WAIST_AVAILABLE else None
                    waist_command, waist_target = waist_sent or (0.0, 0.0)
                    actions = {
                        "left_arm": {
                            "qpos":   left_arm_action.tolist(),
                            "qvel":   [],
                            "torque": left_arm_tauff.tolist(),
                        },
                        "right_arm": {
                            "qpos":   right_arm_action.tolist(),
                            "qvel":   [],
                            "torque": right_arm_tauff.tolist(),
                        },
                        "left_ee": {
                            "qpos":   [],
                            "qvel":   [],
                            "torque": [],
                            # The pre-IK target: the only EE quantity here that cannot
                            # be recomputed from the joint channels afterwards.
                            "target_pose":     ee_kinematics.mat_to_pose_vec(left_wrist_target),
                            "target_pose_mat": ee_kinematics.mat_to_flat(left_wrist_target),
                            **_pose_fields(ee_action_pose["left"], args.record_ee_pose),
                        },
                        "right_ee": {
                            "qpos":   [],
                            "qvel":   [],
                            "torque": [],
                            "target_pose":     ee_kinematics.mat_to_pose_vec(right_wrist_target),
                            "target_pose_mat": ee_kinematics.mat_to_flat(right_wrist_target),
                            **_pose_fields(ee_action_pose["right"], args.record_ee_pose),
                        },
                        # Kept at its historical key and meaning ([vx, vy, wz]) so
                        # existing converter configs keep resolving, but now always
                        # length 3: "no command" is recorded as zeros rather than as a
                        # missing field, otherwise manipulation-only episodes and
                        # locomotion episodes cannot share one dataset.
                        "body": {
                            "qpos": cmd_vel,
                        },
                        "base": {
                            "cmd_vel":     cmd_vel,
                            "cmd_vel_sent": bool(cmd_vel_sent),
                            "stick_left":  left_stick,
                            "stick_right": right_stick,
                        },
                        "ik_residual": ik_residual or {},
                        "waist": {
                            "qpos":   [float(waist_command), 0.0, 0.0],
                            "target": [float(waist_target), 0.0, 0.0],
                        },
                    }
                    if args.record_wireless_remote and low is not None and "wireless_remote" in low:
                        actions["base"]["wireless_remote"] = low["wireless_remote"]

                    # Per-camera capture time from the TIM2 header, the time the action
                    # was solved, and how old each DDS sample was at that moment.
                    # rt/odommodestate publishes stamp=0 on G1, so receive time is the
                    # only clock available for it.
                    timestamps = {"action_wall_time_us": action_wall_time_us}
                    for cam_key, cam_frame in (("head_camera", head_img),
                                               ("left_wrist_camera", left_wrist_img),
                                               ("right_wrist_camera", right_wrist_img)):
                        camera_timestamp = recording_camera_timestamp(
                            cam_frame, action_wall_time_us
                        )
                        if camera_timestamp is not None:
                            timestamps[cam_key] = camera_timestamp
                    for probe_key, probe_snap in probe_data.items():
                        if probe_snap is not None:
                            timestamps[probe_key] = {
                                "recv_wall_time_us": probe_snap["recv_wall_time_us"],
                                "age_ms": probe_snap["age_ms"],
                            }

                    # Read once, so every field of this frame's annotation agrees
                    # even if the labelling operator presses a key mid-iteration.
                    label_id = LABEL_ID
                    annotation = {"label_id": label_id, "label": LABEL_NAMES.get(label_id, "")}

                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions,
                                          sim_state=sim_state, timestamps=timestamps,
                                          annotation=annotation)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions,
                                          timestamps=timestamps, annotation=annotation)

            sleep_time = loop_rate.sleep()
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        try:
            go_home_and_release(arm_ctrl)
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")

        try:
            if args.ipc:
                ipc_server.stop()
            else:
                KEYBOARD_STOP_EVENT.set()
                if listen_keyboard_thread.is_alive():
                    listen_keyboard_thread.join(timeout=1.0)
        except Exception as e:
                logger_mp.error(f"Failed to stop keyboard listener or IPC server: {e}")

        try:
            if probes is not None:
                probes.close()
        except Exception as e:
            logger_mp.error(f"Failed to close robot state probes: {e}")

        try:
            if img_client is not None:
                img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            if rec_img_client is not None:
                logger_mp.info(f"Recording stream stats: {rec_img_client.stats()}")
                rec_img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recording image client: {e}")

        try:
            if xr_notifier is not None:
                # Leave nothing pinned in the scene for the next run to inherit.
                xr_notifier.clear()
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")

        try:
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
