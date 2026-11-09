import numpy as np
import threading
import time
import copy
from enum import IntEnum

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
    IMUState_ as hg_IMUState, LowCmd_ as hg_LowCmd, LowState_ as hg_LowState) # idl for g1, h1_2
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

from unitree_sdk2py.idl.unitree_go.msg.dds_ import ( LowCmd_  as go_LowCmd, LowState_ as go_LowState)  # idl for h1
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

from teleop.robot_control.waist_yaw import WAIST_ENABLE_ID, WAIST_YAW_ID, WaistYawRamp

kTopicLowCommand_Debug  = "rt/lowcmd"
kTopicLowCommand_Motion = "rt/arm_sdk"
kTopicLowState = "rt/lowstate"
kTopicSecondaryImu = "rt/secondary_imu"

G1_29_Num_Motors = 35
G1_23_Num_Motors = 35
H1_2_Num_Motors = 35
H1_Num_Motors = 20
H2_Num_Motors = 35
 

class MotorState:
    def __init__(self):
        self.q = None
        self.dq = None
        self.tau_est = 0.0
        self.motorstate = None
        self.temperature = None

class G1_29_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_29_Num_Motors)]
        self.imu_quat = [1.0, 0.0, 0.0, 0.0]
        self.imu_gyro = [0.0] * 3
        self.imu_accel = [0.0] * 3
        self.torso_imu_quat = [1.0, 0.0, 0.0, 0.0]
        self.torso_imu_gyro = [0.0] * 3
        self.torso_imu_accel = [0.0] * 3
        self.torso_received_monotonic = 0.0
        self.mode_machine = 0
        self.sequence = 0
        self.received_monotonic = 0.0

class G1_23_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_23_Num_Motors)]

class H1_2_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(H1_2_Num_Motors)]

class H1_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(H1_Num_Motors)]

class H2_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(H2_Num_Motors)]


class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data

class G1_29_ArmController:
    def __init__(self, motion_mode = False, simulation_mode = False, body_controller=None,
                 validate_lowstate_crc=None, home_q=None, independent_joint_limits=False,
                 motor_health_guard=False, smooth_home=False, home_duration_s=3.0,
                 home_max_velocity=0.6, home_max_acceleration=1.0, home_tauff=None):
        if body_controller is not None and motion_mode:
            raise ValueError("A body controller must own rt/lowcmd; motion_mode cannot be combined with it")
        self._home_q = np.zeros(14) if home_q is None else np.array(home_q, dtype=float, copy=True)
        if self._home_q.shape != (14,) or not np.all(np.isfinite(self._home_q)):
            raise ValueError("G1_29 home_q must contain 14 finite joint angles in radians")
        self._home_tauff = np.zeros(14) if home_tauff is None else np.array(home_tauff, dtype=float, copy=True)
        if self._home_tauff.shape != (14,) or not np.all(np.isfinite(self._home_tauff)):
            raise ValueError("G1_29 home_tauff must contain 14 finite joint efforts")
        from teleop.utils.home_trajectory import home_trajectory_duration
        home_trajectory_duration(self._home_q, self._home_q, home_duration_s,
                                 home_max_velocity, home_max_acceleration)
        self.smooth_home = bool(smooth_home) and body_controller is None
        self.home_duration_s = float(home_duration_s)
        self.home_max_velocity = float(home_max_velocity)
        self.home_max_acceleration = float(home_max_acceleration)
        self._home_plan = None
        self._home_source = None
        logger_mp.info("Initialize G1_29_ArmController...")
        self.body_controller = body_controller
        self._running = True
        self._last_command = None
        self.ctrl_lock = threading.Lock()
        self._motor_health_guard = None
        if motor_health_guard and body_controller is None and not simulation_mode:
            from teleop.utils.arm_motor_health import ArmMotorHealthGuard
            self._motor_health_guard = ArmMotorHealthGuard()
        self.independent_joint_limits = bool(independent_joint_limits)
        self._command_sequence = 0
        self._command_monotonic = time.monotonic()
        self._native_control_warning = None
        self._warning_last_log = {}
        self._last_finite_arm_q = None
        self._native_measurement_valid = False
        self._last_valid_target_q = self._home_q.copy()
        self._last_valid_target_tauff = np.zeros(14)
        self._last_successful_native_command = None
        # Set the resting target before the motor writer starts, so startup and
        # later homing use the same pose without a brief move towards zero.
        self.q_target = self._home_q.copy()
        self.tauff_target = np.zeros(14)
        self.motion_mode = motion_mode
        self.simulation_mode = simulation_mode
        self.kp_high = 300.0
        self.kd_high = 3.0
        self.kp_low = 80.0
        self.kd_low = 3.0
        self.kp_wrist = 40.0
        self.kd_wrist = 1.5

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None
        self.crc = CRC()
        if validate_lowstate_crc is None:
            validate_lowstate_crc = body_controller is not None and not simulation_mode
        self.validate_lowstate_crc = bool(validate_lowstate_crc)
        self.lowstate_crc = None
        if self.validate_lowstate_crc:
            from teleop.robot_control.sonic.crc import HGLowStateCRC
            self.lowstate_crc = HGLowStateCRC()
        self._lowstate_crc_errors = 0

        self.lowcmd_publisher = None
        self.lowstate_subscriber = None
        self.imutorso_subscriber = None
        self.subscribe_thread = None
        self.publish_thread = None
        self._subscribe_started = False
        self._publish_started = False
        try:
            self._initialize_controller_io()
        except BaseException:
            # __init__ has not returned, so its caller cannot close partially
            # initialized DDS endpoints or a blocked reader for us.
            self._running = False
            self._join_io_threads()
            self._close_io_channels()
            raise

    def _initialize_controller_io(self):
        if self.motion_mode:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
        else:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.imutorso_subscriber = None
        self._torso_lock = threading.Lock()
        self._torso_imu = None
        if self.body_controller is not None:
            self.imutorso_subscriber = ChannelSubscriber(kTopicSecondaryImu, hg_IMUState)
            self.imutorso_subscriber.Init(self._update_torso_imu, 1)
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state, name="g1-lowstate")
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()
        self._subscribe_started = True

        state_deadline = time.monotonic() + 10.0
        while not self.lowstate_buffer.GetData():
            if time.monotonic() >= state_deadline:
                raise RuntimeError("G1 DDS LowState startup timed out")
            time.sleep(0.1)
            logger_mp.warning("[G1_29_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[G1_29_ArmController] Subscribe dds ok.")

        initial_motor_q = None
        while self.body_controller is None:
            state = self.lowstate_buffer.GetData()
            self._check_motor_health(state, time.monotonic())
            try:
                initial_motor_q = np.asarray([state.motor_state[id].q for id in G1_29_JointIndex], dtype=float)
            except (AttributeError, IndexError, TypeError, ValueError):
                initial_motor_q = None
            if (self._native_measurement_valid and initial_motor_q is not None
                    and np.all(np.isfinite(initial_motor_q))):
                break
            if time.monotonic() >= state_deadline:
                raise RuntimeError("G1 DDS LowState startup has no finite initial joint positions")
            time.sleep(.02)

        # initialize hg's lowcmd msg
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = initial_motor_q if self.body_controller is None else self.get_current_motor_q()
        if self.body_controller is not None:
            # The mixed controller starts publishing before its parent can send
            # the first IK result. Hold measured arms instead of briefly aiming
            # the default zero vector during that startup window.
            self.q_target = self.get_current_dual_arm_q().copy()
        logger_mp.debug(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.debug(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")

        arm_indices = set(member.value for member in G1_29_JointArmIndex)
        for id in G1_29_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]
        # A waist turn is a request to DreamWaQ on rt/arm_sdk (waist_yaw.py), so
        # only motion mode has one; in debug mode slot 12 drives the motor itself.
        self.waist_yaw = WaistYawRamp() if self.motion_mode and self.body_controller is None else None
        if self.waist_yaw is not None:
            self.msg.motor_cmd[WAIST_YAW_ID].q = 0.0
            self.msg.motor_cmd[WAIST_ENABLE_ID].q = 0.0
        logger_mp.info("Lock OK!")
        self._arm_gains = [(self.msg.motor_cmd[id].kp, self.msg.motor_cmd[id].kd)
                           for id in G1_29_JointArmIndex]
        if self.smooth_home:
            # Construct the fixed measured-to-home plan before the first writer
            # tick. Its first q and effort are the measured hold and zero FF.
            self.request_home()

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state, name="g1-lowcmd")
        if self.body_controller is not None:
            self.control_dt = 1.0 / 500.0
            self.body_controller.start(self.lowstate_buffer.GetData)
        self.publish_thread.daemon = True
        self.publish_thread.start()
        self._publish_started = True

        logger_mp.info("Initialize G1_29_ArmController OK!")

    def _subscribe_motor_state(self):
        sequence = 0
        while self._running:
            try:
                # A positive timeout is required: SDK None/0 means an
                # effectively infinite DDS take_one wait, even during close.
                msg = self.lowstate_subscriber.Read(timeout=0.02)
            except Exception as error:
                if not self._running:
                    return  # Closing the reader intentionally unblocks Read().
                if self.body_controller is not None:
                    raise
                self._set_native_control_warning('lowstate_read_failed', None, time.monotonic(), str(error))
                time.sleep(.02)
                continue
            if msg is not None:
                if self.validate_lowstate_crc:
                    crc_error = None
                    try:
                        crc_valid = int(msg.crc) == self.lowstate_crc.Crc(msg)
                    except Exception as error:
                        crc_valid = False
                        crc_error = str(error)
                    if not crc_valid:
                        self._lowstate_crc_errors += 1
                        if self._lowstate_crc_errors <= 5 or self._lowstate_crc_errors % 100 == 0:
                            detail = "mismatch" if crc_error is None else f"validation failed: {crc_error}"
                            logger_mp.error(f"G1 LowState CRC {detail} "
                                            f"({self._lowstate_crc_errors} rejected)")
                        time.sleep(0.002)
                        continue
                lowstate = G1_29_LowState()
                for id in range(G1_29_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                    lowstate.motor_state[id].tau_est = getattr(msg.motor_state[id], 'tau_est', 0.0)
                    lowstate.motor_state[id].motorstate = getattr(msg.motor_state[id], 'motorstate', None)
                    temperature = getattr(msg.motor_state[id], 'temperature', None)
                    try:
                        lowstate.motor_state[id].temperature = list(temperature)
                    except TypeError:
                        lowstate.motor_state[id].temperature = None
                lowstate.imu_quat = list(msg.imu_state.quaternion)
                lowstate.imu_gyro = list(msg.imu_state.gyroscope)
                lowstate.imu_accel = list(msg.imu_state.accelerometer)
                with self._torso_lock:
                    torso = self._torso_imu
                if torso is not None:
                    lowstate.torso_imu_quat = torso[0].copy()
                    lowstate.torso_imu_gyro = torso[1].copy()
                    lowstate.torso_imu_accel = torso[2].copy()
                    lowstate.torso_received_monotonic = torso[3]
                lowstate.mode_machine = msg.mode_machine
                lowstate.sequence = sequence
                lowstate.received_monotonic = time.monotonic()
                sequence += 1
                self.lowstate_buffer.SetData(lowstate)
                self._clear_native_control_warning('lowstate_read_failed')
            time.sleep(0.002)

    def _update_torso_imu(self, msg):
        candidate = [*msg.quaternion, *msg.gyroscope, *msg.accelerometer]
        if len(candidate) != 10 or not np.all(np.isfinite(candidate)):
            return
        with self._torso_lock:
            self._torso_imu = (list(msg.quaternion), list(msg.gyroscope),
                               list(msg.accelerometer), time.monotonic())

    def clip_arm_q_target(self, target_q, velocity_limit, current_q=None):
        if current_q is None:
            current_q = self.get_current_dual_arm_q()
        if self.independent_joint_limits and self.body_controller is None:
            from teleop.utils.arm_joint_limits import clip_joint_targets
            return clip_joint_targets(target_q, current_q, velocity_limit, self.control_dt)
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        from teleop.utils.loop_rate import LoopRate
        rate = LoopRate(1.0 / self.control_dt, clock=time.monotonic, sleep=time.sleep)
        sequence, previous_tick, max_interval_ms = 0, None, 0.0
        if self.motion_mode:
            self.msg.motor_cmd[G1_29_JointIndex.kNotUsedJoint0].q = 1.0;

        while self._running:
            tick = time.monotonic()
            writer_interval_ms = None if previous_tick is None else (tick - previous_tick) * 1000
            loop_dt = 0.0 if previous_tick is None else min(tick - previous_tick, 0.1)
            if previous_tick is not None:
                max_interval_ms = max(max_interval_ms, (tick - previous_tick) * 1000)
            previous_tick = tick

            with self.ctrl_lock:
                home_plan = getattr(self, '_home_plan', None)
                home_source = getattr(self, '_home_source', None)
                if home_plan is not None:
                    self.q_target, self.tauff_target = home_plan.sample(tick)
                arm_q_target     = np.array(self.q_target, copy=True)
                arm_tauff_target = np.array(self.tauff_target, copy=True)
                command_sequence = self._command_sequence
                command_monotonic = self._command_monotonic
                if self.waist_yaw is not None:
                    waist_command = self.waist_yaw.step(loop_dt)
                    waist_active = self.waist_yaw.active
            if self.waist_yaw is not None:
                self.msg.motor_cmd[WAIST_YAW_ID].q = waist_command
                self.msg.motor_cmd[WAIST_YAW_ID].dq = 0.0
                self.msg.motor_cmd[WAIST_ENABLE_ID].q = 1.0 if waist_active else 0.0

            if self.body_controller is not None:
                if (np.shape(arm_q_target) != (14,) or np.shape(arm_tauff_target) != (14,)
                        or not np.all(np.isfinite(arm_q_target)) or not np.all(np.isfinite(arm_tauff_target))):
                    self.body_controller.trip("Invalid XR IK arm target or torque")
                    arm_q_target = self.get_current_dual_arm_q()
                    arm_tauff_target = np.zeros(14)

            lowstate = None
            measured_arm_q = None
            motor_health_warning = None
            replay_command = None
            publish_allowed, skipped_reason = True, None
            if self.body_controller is None:
                lowstate = self.lowstate_buffer.GetData()
                motor_health_warning = self._check_motor_health(lowstate, time.monotonic())
                measured_arm_q = self._finite_native_hold_q()
                if (self._native_arm_vector(arm_q_target) is not None
                        and self._native_arm_vector(arm_tauff_target) is not None):
                    self._last_valid_target_q = arm_q_target.copy()
                    self._last_valid_target_tauff = arm_tauff_target.copy()
                else:
                    self._set_native_control_warning('invalid_arm_command', lowstate, tick)
                    arm_q_target = self._native_arm_vector(getattr(self, '_last_valid_target_q', None))
                    arm_tauff_target = self._native_arm_vector(getattr(self, '_last_valid_target_tauff', None))
                    if arm_q_target is None or arm_tauff_target is None:
                        publish_allowed, skipped_reason = False, 'no_valid_arm_command'
                if not self._native_measurement_valid:
                    replay_command = self._native_replay_command()
                    publish_allowed = replay_command is not None
                    skipped_reason = None if publish_allowed else 'invalid_measured_arm_state'

            if replay_command is not None:
                cliped_arm_q_target = np.array(replay_command['qpos'], copy=True)
                arm_tauff_target = np.array(replay_command['torque'], copy=True)
            elif not publish_allowed:
                # Bad numeric data is omitted, never turned into damping or
                # an invented pose. The last delivered command is untouched.
                cliped_arm_q_target = None
            elif self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(
                    arm_q_target, velocity_limit=self.arm_velocity_limit, current_q=measured_arm_q)

            body_valid = True
            if self.body_controller is not None:
                body_valid = self.body_controller.apply(self.msg, self.lowstate_buffer.GetData())
            if body_valid and publish_allowed:
                for idx, id in enumerate(G1_29_JointArmIndex):
                    if self.body_controller is not None:
                        # Startup damping clears gains; restore the existing IK PD profile.
                        self.msg.motor_cmd[id].kp, self.msg.motor_cmd[id].kd = self._arm_gains[idx]
                    elif replay_command is not None:
                        self.msg.motor_cmd[id].kp = replay_command['kp'][idx]
                        self.msg.motor_cmd[id].kd = replay_command['kd'][idx]
                    self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                    self.msg.motor_cmd[id].dq = replay_command['qvel'][idx] if replay_command is not None else 0
                    self.msg.motor_cmd[id].tau = arm_tauff_target[idx]

            published = False
            write_error = None
            if self.body_controller is not None:
                self.msg.crc = self.crc.Crc(self.msg)
            elif publish_allowed:
                try:
                    self.msg.crc = self.crc.Crc(self.msg)
                    self._clear_native_control_warning('crc_seal_failed')
                except Exception as error:
                    # Never emit a packet whose CRC could not be computed.
                    # A later tick can retry without ending the writer thread.
                    write_error = error
                    publish_allowed, skipped_reason = False, 'crc_seal_failed'
                    self._set_native_control_warning('crc_seal_failed', lowstate,
                                                     time.monotonic(), str(error))
            if publish_allowed and (self.body_controller is None or not self.body_controller.dry_run):
                try:
                    published = self.lowcmd_publisher.Write(self.msg) is not False
                except Exception as error:
                    if self.body_controller is None:
                        write_error = error
                    else:
                        self.body_controller.trip(f"DDS LowCmd publication failed: {error}")
                if self.body_controller is not None and not published:
                    self.body_controller.trip("DDS LowCmd publication failed")
                    body_valid = False
                elif self.body_controller is None and not published:
                    self._set_native_control_warning('dds_publish_failed', lowstate, time.monotonic(),
                                                     str(write_error) if write_error is not None else None)
                elif self.body_controller is None:
                    self._clear_native_control_warning('dds_publish_failed')
                    sent = {name: [float(getattr(motor, field)) for motor in self.msg.motor_cmd[15:29]]
                            for name, field in (('qpos', 'q'), ('qvel', 'dq'), ('torque', 'tau'),
                                                ('kp', 'kp'), ('kd', 'kd'))}
                    with self.ctrl_lock:
                        self._last_successful_native_command = sent
            if self.body_controller is not None:
                with self.ctrl_lock:
                    self._last_command = {
                        name: [float(getattr(m, attr)) for m in self.msg.motor_cmd[:29]]
                        for name, attr in (('qpos', 'q'), ('qvel', 'dq'), ('torque', 'tau'),
                                           ('kp', 'kp'), ('kd', 'kd'))
                    }
                    self._last_command.update(
                        publish_wall_time_us=time.time_ns() // 1000, valid=body_valid,
                        publish_monotonic=tick, sequence=sequence,
                        max_writer_interval_ms=max_interval_ms,
                        published=published,
                        sonic=self.body_controller.applied_snapshot())
            else:
                published_at = time.monotonic()
                command = dict(
                    arm_qpos=cliped_arm_q_target.tolist() if cliped_arm_q_target is not None else None,
                    arm_target_qpos=arm_q_target.tolist() if self._native_arm_vector(arm_q_target) is not None else None,
                    arm_torque=arm_tauff_target.tolist() if arm_tauff_target is not None else None,
                    arm_measured_qpos=measured_arm_q.tolist() if self._native_measurement_valid else None,
                    arm_hold_qpos=measured_arm_q.tolist() if measured_arm_q is not None else None,
                    measurement_valid=self._native_measurement_valid,
                    replayed_last_command=replay_command is not None,
                    command_sequence=command_sequence, command_monotonic=command_monotonic,
                    publish_wall_time_us=time.time_ns() // 1000, publish_monotonic=published_at,
                    sequence=sequence, writer_interval_ms=writer_interval_ms,
                    max_writer_interval_ms=max_interval_ms, published=published,
                    valid=bool(publish_allowed and published),
                    lowstate_sequence=getattr(lowstate, 'sequence', None),
                    lowstate_received_monotonic=getattr(lowstate, 'received_monotonic', None),
                    motor_health=dict(enabled=getattr(self, '_motor_health_guard', None) is not None,
                                      mode='warning_only', fault=None,
                                      warning=self.get_motor_health_warning()),
                    skipped_reason=skipped_reason,
                    home=self._describe_home(home_plan, home_source, command_sequence,
                                             measured_arm_q, tick, None)
                         if getattr(self, 'smooth_home', False) else None,
                )
                if write_error is not None:
                    command['publish_error'] = str(write_error)
                with self.ctrl_lock:
                    self._last_command = command
            # Native publication errors are observations only. Retry normal
            # commands without altering gains, effort, or control authority.
            sequence += 1

            if self._speed_gradual_max is True:
                t_elapsed = tick - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            rate.sleep()
            # logger_mp.debug(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # logger_mp.debug(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        if getattr(self, 'body_controller', None) is None:
            q, effort = self._native_arm_vector(q_target), self._native_arm_vector(tauff_target)
            if q is None or effort is None:
                self._set_native_control_warning('invalid_arm_command', None, time.monotonic())
                return None  # Keep the previous valid target and any home plan.
            q_target, tauff_target = q, effort
            self._clear_native_control_warning('invalid_arm_command')
        with self.ctrl_lock:
            if getattr(self, '_home_plan', None) is not None:
                self._home_source = 'cancelled_by_tracking'
            self._home_plan = None
            self.q_target = np.array(q_target, copy=True)
            self.tauff_target = np.array(tauff_target, copy=True)
            if getattr(self, 'body_controller', None) is None:
                self._last_valid_target_q = self.q_target.copy()
                self._last_valid_target_tauff = self.tauff_target.copy()
            self._command_sequence += 1
            self._command_monotonic = time.monotonic()
            return self._command_sequence

    def set_waist_yaw_target(self, target_rad):
        '''Ask DreamWaQ to hold the waist yaw at target_rad (0 = default pose).'''
        if self.waist_yaw is None:
            raise RuntimeError("a waist turn needs --motion: DreamWaQ owns the waist")
        with self.ctrl_lock:
            self.waist_yaw.set_target(target_rad)

    def get_waist_yaw(self):
        '''(sent command, target) in rad, or None when this controller cannot turn the waist.'''
        if self.waist_yaw is None:
            return None
        with self.ctrl_lock:
            return self.waist_yaw.command, self.waist_yaw.target

    def wait_waist_yaw_settled(self, timeout=3.0):
        '''Wait until the sent waist command has reached its target.'''
        deadline = time.monotonic() + timeout
        while self.waist_yaw is not None and time.monotonic() < deadline:
            with self.ctrl_lock:
                if self.waist_yaw.settled:
                    return True
            time.sleep(0.02)
        return self.waist_yaw is None

    def get_mode_machine(self):
        '''Return current dds mode machine.'''
        return self.lowstate_buffer.GetData().mode_machine

    def get_last_command(self):
        """Atomic copy of the last motor-thread write attempt, including its outcome."""
        with self.ctrl_lock:
            return copy.deepcopy(getattr(self, '_last_command', None))

    @staticmethod
    def _native_arm_vector(value):
        try:
            array = np.asarray(value, dtype=float)
            if array.shape == (14,) and np.all(np.isfinite(array)):
                return array
        except (TypeError, ValueError, OverflowError):
            pass
        return None

    def _finite_native_hold_q(self):
        """Return a known finite observation or sent position, without inventing one."""
        known = self._native_arm_vector(getattr(self, '_last_finite_arm_q', None))
        if known is not None:
            return known.copy()
        command = self.get_last_command()
        if command is not None and command.get('published'):
            known = self._native_arm_vector(command.get('arm_qpos'))
            if known is not None:
                return known.copy()
        return None

    def _native_replay_command(self):
        """Return only an actual successful finite arm packet for reuse."""
        with self.ctrl_lock:
            command = copy.deepcopy(getattr(self, '_last_successful_native_command', None))
        if not isinstance(command, dict):
            return None
        if any(self._native_arm_vector(command.get(key)) is None
               for key in ('qpos', 'qvel', 'torque', 'kp', 'kd')):
            return None
        return command

    def _set_native_control_warning(self, reason, state, now_monotonic, detail=None):
        with self.ctrl_lock:
            warning = dict(
                message=f"G1 native arm warning: {reason}; keeping control active",
                kind='native_control', reason=reason, detected_monotonic=float(now_monotonic),
                lowstate_sequence=getattr(state, 'sequence', None),
                lowstate_received_monotonic=getattr(state, 'received_monotonic', None),
                issues=[dict(reasons=[reason])],
            )
            if detail:
                warning['detail'] = str(detail)[:240]
            self._native_control_warning = warning
        self._log_motor_warning(warning, now_monotonic)
        return copy.deepcopy(warning)

    def _clear_native_control_warning(self, reason):
        with self.ctrl_lock:
            current = getattr(self, '_native_control_warning', None)
            if current is not None and current.get('reason') == reason:
                self._native_control_warning = None

    def _log_motor_warning(self, warning, now_monotonic):
        if warning is None:
            return
        key = tuple((issue.get('motor_index'), tuple(issue.get('reasons', [])))
                    for issue in warning.get('issues', []))
        with self.ctrl_lock:
            stamps = getattr(self, '_warning_last_log', None)
            if stamps is None:
                self._warning_last_log = stamps = {}
            previous = stamps.get(key, float('-inf'))
            if now_monotonic - previous < 5.:
                return
            stamps[key] = now_monotonic
        logger_mp.warning(warning['message'])

    def _check_motor_health(self, state, now_monotonic):
        if self.body_controller is None:
            try:
                q = self._native_arm_vector([state.motor_state[id].q for id in G1_29_JointArmIndex])
                dq = self._native_arm_vector([state.motor_state[id].dq for id in G1_29_JointArmIndex])
            except (AttributeError, IndexError, TypeError):
                q = dq = None
            self._native_measurement_valid = q is not None and dq is not None
            if not self._native_measurement_valid:
                self._set_native_control_warning('invalid_measured_arm_state', state, now_monotonic)
            else:
                self._last_finite_arm_q = q.copy()
                self._clear_native_control_warning('invalid_measured_arm_state')
        guard = getattr(self, '_motor_health_guard', None)
        if guard is None:
            return None
        with self.ctrl_lock:
            warning = guard.check(state, now_monotonic)
        self._log_motor_warning(warning, now_monotonic)
        return warning

    def get_motor_health_fault(self):
        """Compatibility API: reported native health never requests a stop."""
        return None

    def get_motor_health_warning(self):
        with self.ctrl_lock:
            warning = getattr(self, '_native_control_warning', None)
            if warning is not None:
                return copy.deepcopy(warning)
            guard = getattr(self, '_motor_health_guard', None)
            return guard.warning() if guard is not None else None

    def _join_io_threads(self):
        for name, started in (('publish_thread', '_publish_started'), ('subscribe_thread', '_subscribe_started')):
            thread = getattr(self, name, None)
            if thread is not None and getattr(self, started, True):
                try:
                    thread.join(timeout=1.0)
                except RuntimeError:
                    pass  # Thread.start() can fail before creating a thread.
                setattr(self, started, False)

    def _close_io_channels(self):
        for name in ('lowstate_subscriber', 'imutorso_subscriber', 'lowcmd_publisher'):
            channel = getattr(self, name, None)
            if channel is not None:
                try:
                    channel.Close()
                except Exception:
                    pass
                setattr(self, name, None)

    def close(self):
        """Stop the optional body backend and flush damping before stopping DDS."""
        if self.body_controller is not None:
            self.body_controller.close()
            # Keep the sole motor writer alive long enough to publish damping.
            time.sleep(0.1)
        self._running = False
        self._join_io_threads()
        self._close_io_channels()
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        state = self.lowstate_buffer.GetData()
        return np.array([state.motor_state[id].q for id in G1_29_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        state = self.lowstate_buffer.GetData()
        return np.array([state.motor_state[id].q for id in G1_29_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        state = self.lowstate_buffer.GetData()
        return np.array([state.motor_state[id].dq for id in G1_29_JointArmIndex])

    def get_current_dual_arm_state(self):
        """Read arm q/dq and body q from one immutable LowState buffer entry."""
        state = self.lowstate_buffer.GetData()
        return dict(q=np.array([state.motor_state[id].q for id in G1_29_JointArmIndex]),
                    dq=np.array([state.motor_state[id].dq for id in G1_29_JointArmIndex]),
                    tau_est=np.array([getattr(state.motor_state[id], 'tau_est', 0.)
                                      for id in G1_29_JointArmIndex]),
                    body_q=np.array([state.motor_state[id].q for id in G1_29_JointIndex]),
                    sequence=state.sequence, received_monotonic=state.received_monotonic,
                    motor_temperature=[copy.deepcopy(getattr(motor, 'temperature', None))
                                       for motor in state.motor_state],
                    motor_status=[getattr(motor, 'motorstate', None) for motor in state.motor_state])

    def get_home_q(self):
        """Return an independent copy of the configured resting arm pose."""
        return self._home_q.copy()

    def get_home_tauff(self):
        """Return an independent copy of the configured home gravity effort."""
        return self._home_tauff.copy()

    def request_home(self):
        """Request a home plan without sleeping; return its remaining seconds.

        Existing home plans, including their completed hold, are reused. A new
        plan starts at the last successfully sent command, or measured q when
        no command has been sent. Tracking cancels this mode explicitly.
        """
        from teleop.utils.home_trajectory import HomeTrajectory
        try:
            measured = self._native_arm_vector(self.get_current_dual_arm_q())
        except (AttributeError, IndexError, TypeError):
            measured = None
        if measured is None:
            measured = self._finite_native_hold_q()
        if measured is None:
            self._set_native_control_warning('home_waiting_for_state', None, time.monotonic())
            return 0.
        now = time.monotonic()
        with self.ctrl_lock:
            plan = getattr(self, '_home_plan', None)
            if plan is not None and np.array_equal(plan.target_q, self._home_q) and np.array_equal(plan.target_tauff, self._home_tauff):
                return plan.progress(now)['remaining_s']
            if not self.smooth_home:
                self._home_plan = None
                self._home_source = 'legacy_direct'
                self.q_target = self._home_q.copy()
                self.tauff_target = self._home_tauff.copy() if self.body_controller is None else np.zeros(14)
                self._command_sequence += 1
                self._command_monotonic = now
                return 0.
            start_q, start_tauff = measured.copy(), np.zeros(14)
            source = 'measured'
            last = self._last_command
            if last is not None and last.get('published') and last.get('valid', True):
                candidate_q = np.asarray(last.get('arm_qpos', []), dtype=float)
                candidate_tauff = np.asarray(last.get('arm_torque', []), dtype=float)
                if (candidate_q.shape == candidate_tauff.shape == (14,)
                        and np.all(np.isfinite(candidate_q)) and np.all(np.isfinite(candidate_tauff))):
                    start_q, start_tauff = candidate_q.copy(), candidate_tauff.copy()
                    source = 'last_published'
            plan = HomeTrajectory(start_q, self._home_q, start_tauff, self._home_tauff, now,
                                  self.home_duration_s, self.home_max_velocity, self.home_max_acceleration)
            self._home_plan, self._home_source = plan, source
            self.q_target, self.tauff_target = plan.sample(now)
            self._command_sequence += 1
            self._command_monotonic = now
            return plan.duration_s

    def _describe_home(self, plan, source, sequence, measured, now, fault):
        progress = (plan.progress(now) if plan is not None else
                    dict(duration_s=0., elapsed_s=0., remaining_s=0., fraction=0., planned_complete=False))
        measured = self._native_arm_vector(measured)
        error = float(np.max(np.abs(measured - self._home_q))) if measured is not None else None
        progress.update(enabled=self.smooth_home, active=plan is not None and not progress['planned_complete'],
                        source=source, command_sequence=sequence, max_error_rad=error,
                        within_tolerance=error is not None and error < .05, aborted=False)
        return progress

    def get_home_progress(self):
        """Distinguish planned completion from actual measured arrival."""
        measured = self.get_current_dual_arm_q()
        now = time.monotonic()
        with self.ctrl_lock:
            return self._describe_home(self._home_plan, self._home_source, self._command_sequence,
                                       measured, now, None)

    def ctrl_dual_arm_go_home(self):
        '''Return home using the configured smooth plan/effort, or legacy direct zero-FF targets.'''
        if getattr(self, 'smooth_home', False):
            self._go_home_smooth()
            return
        logger_mp.info("[G1_29_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        home_q = self.get_home_q()
        with self.ctrl_lock:
            self.q_target = home_q.copy()
            self.tauff_target = np.zeros(14)
            self._command_sequence += 1
            self._command_monotonic = time.monotonic()
            if self.waist_yaw is not None:
                self.waist_yaw.set_target(0.0)
        tolerance = 0.05  # Maximum joint error from the configured home pose, radians.
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q - home_q) < tolerance):
                if self.motion_mode:
                    # Unwind a waist turn before the weight drops: below 0.5
                    # DreamWaQ stops reading the request and unwinds on its own.
                    if not self.wait_waist_yaw_settled():
                        logger_mp.warning("[G1_29_ArmController] waist turn did not unwind in time; "
                                          "DreamWaQ returns it to default once the weight drops.")
                    for weight in np.linspace(1, 0, num=101):
                        with self.ctrl_lock:
                            self.msg.motor_cmd[G1_29_JointIndex.kNotUsedJoint0].q = weight;
                        time.sleep(0.02)
                logger_mp.info("[G1_29_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def _go_home_smooth(self):
        remaining = self.request_home()
        logger_mp.info("[G1_29_ArmController] smooth home: %.2fs planned + 5s settling", remaining)
        deadline = time.monotonic() + remaining + 5.
        while time.monotonic() < deadline:
            progress = self.get_home_progress()
            if progress['planned_complete'] and progress['within_tolerance']:
                if self.motion_mode:
                    with self.ctrl_lock:
                        if self.waist_yaw is not None:
                            self.waist_yaw.set_target(0.)
                    if not self.wait_waist_yaw_settled():
                        logger_mp.warning("[G1_29_ArmController] waist turn did not unwind in time.")
                    for weight in np.linspace(1., 0., num=101):
                        with self.ctrl_lock:
                            self.msg.motor_cmd[G1_29_JointIndex.kNotUsedJoint0].q = weight
                        time.sleep(.02)
                logger_mp.info("[G1_29_ArmController] smooth home plan and measured arrival confirmed.")
                return
            time.sleep(.05)
        error = self.get_home_progress()['max_error_rad']
        logger_mp.warning("[G1_29_ArmController] smooth home not confirmed: max error %s rad; retaining home hold.",
                          f"{error:.4f}" if error is not None else 'unavailable')

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.monotonic()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            G1_29_JointIndex.kLeftAnklePitch.value,
            G1_29_JointIndex.kRightAnklePitch.value,
            # Left arm
            G1_29_JointIndex.kLeftShoulderPitch.value,
            G1_29_JointIndex.kLeftShoulderRoll.value,
            G1_29_JointIndex.kLeftShoulderYaw.value,
            G1_29_JointIndex.kLeftElbow.value,
            # Right arm
            G1_29_JointIndex.kRightShoulderPitch.value,
            G1_29_JointIndex.kRightShoulderRoll.value,
            G1_29_JointIndex.kRightShoulderYaw.value,
            G1_29_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors
    
    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            G1_29_JointIndex.kLeftWristRoll.value,
            G1_29_JointIndex.kLeftWristPitch.value,
            G1_29_JointIndex.kLeftWristyaw.value,
            G1_29_JointIndex.kRightWristRoll.value,
            G1_29_JointIndex.kRightWristPitch.value,
            G1_29_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors

class G1_29_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

class G1_29_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28
    
    # not used
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34

class G1_23_ArmController:
    def __init__(self, motion_mode = False, simulation_mode = False):
        self.simulation_mode = simulation_mode
        self.motion_mode = motion_mode

        logger_mp.info("Initialize G1_23_ArmController...")
        self.q_target = np.zeros(10)
        self.tauff_target = np.zeros(10)

        self.kp_high = 300.0
        self.kd_high = 3.0
        self.kp_low = 80.0
        self.kd_low = 3.0
        self.kp_wrist = 40.0
        self.kd_wrist = 1.5

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

        
        if self.motion_mode:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
        else:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[G1_23_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[G1_23_ArmController] Subscribe dds ok.")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")

        arm_indices = set(member.value for member in G1_23_JointArmIndex)
        for id in G1_23_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]
        logger_mp.info("Lock OK!")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize G1_23_ArmController OK!")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_23_LowState()
                for id in range(G1_23_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        if self.motion_mode:
            self.msg.motor_cmd[G1_23_JointIndex.kNotUsedJoint0].q = 1.0;

        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target     = self.q_target
                arm_tauff_target = self.tauff_target

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            for idx, id in enumerate(G1_23_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]      

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # logger_mp.debug(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # logger_mp.debug(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target

    def get_mode_machine(self):
        '''Return current dds mode machine.'''
        return self.lowstate_subscriber.Read().mode_machine
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_23_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_23_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in G1_23_JointArmIndex])
    
    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        logger_mp.info("[G1_23_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(10)
            # self.tauff_target = np.zeros(10)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                if self.motion_mode:
                    for weight in np.linspace(1, 0, num=101):
                        self.msg.motor_cmd[G1_23_JointIndex.kNotUsedJoint0].q = weight;
                        time.sleep(0.02)
                logger_mp.info("[G1_23_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            G1_23_JointIndex.kLeftAnklePitch.value,
            G1_23_JointIndex.kRightAnklePitch.value,
            # Left arm
            G1_23_JointIndex.kLeftShoulderPitch.value,
            G1_23_JointIndex.kLeftShoulderRoll.value,
            G1_23_JointIndex.kLeftShoulderYaw.value,
            G1_23_JointIndex.kLeftElbow.value,
            # Right arm
            G1_23_JointIndex.kRightShoulderPitch.value,
            G1_23_JointIndex.kRightShoulderRoll.value,
            G1_23_JointIndex.kRightShoulderYaw.value,
            G1_23_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors
    
    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            G1_23_JointIndex.kLeftWristRoll.value,
            G1_23_JointIndex.kRightWristRoll.value,
        ]
        return motor_index.value in wrist_motors

class G1_23_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26

class G1_23_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRollNotUsed = 13
    kWaistPitchNotUsed = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitchNotUsed = 20
    kLeftWristyawNotUsed = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitchNotUsed = 27
    kRightWristYawNotUsed = 28
    
    # not used
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34

class H1_2_ArmController:
    def __init__(self, motion_mode = False, simulation_mode = False):
        self.simulation_mode = simulation_mode
        self.motion_mode = motion_mode
        
        logger_mp.info("Initialize H1_2_ArmController...")
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)

        self.kp_high = 300.0
        self.kd_high = 5.0
        self.kp_low = 140.0
        self.kd_low = 3.0
        self.kp_wrist = 50.0
        self.kd_wrist = 2.0

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None


        if self.motion_mode:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
        else:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[H1_2_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[H1_2_ArmController] Subscribe dds ok.")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")

        arm_indices = set(member.value for member in H1_2_JointArmIndex)
        for id in H1_2_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]
        logger_mp.info("Lock OK!")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize H1_2_ArmController OK!")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = H1_2_LowState()
                for id in range(H1_2_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        if self.motion_mode:
            self.msg.motor_cmd[H1_2_JointIndex.kNotUsedJoint0].q = 1.0;

        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target     = self.q_target
                arm_tauff_target = self.tauff_target

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            for idx, id in enumerate(H1_2_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]      

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # logger_mp.debug(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # logger_mp.debug(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target

    def get_mode_machine(self):
        '''Return current dds mode machine.'''
        return self.lowstate_subscriber.Read().mode_machine
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H1_2_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H1_2_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in H1_2_JointArmIndex])
    
    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        logger_mp.info("[H1_2_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(14)
            # self.tauff_target = np.zeros(14)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                if self.motion_mode:
                    for weight in np.linspace(1, 0, num=101):
                        self.msg.motor_cmd[H1_2_JointIndex.kNotUsedJoint0].q = weight;
                        time.sleep(0.02)
                logger_mp.info("[H1_2_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            H1_2_JointIndex.kLeftAnkle.value,
            H1_2_JointIndex.kRightAnkle.value,
            # Left arm
            H1_2_JointIndex.kLeftShoulderPitch.value,
            H1_2_JointIndex.kLeftShoulderRoll.value,
            H1_2_JointIndex.kLeftShoulderYaw.value,
            H1_2_JointIndex.kLeftElbowPitch.value,
            # Right arm
            H1_2_JointIndex.kRightShoulderPitch.value,
            H1_2_JointIndex.kRightShoulderRoll.value,
            H1_2_JointIndex.kRightShoulderYaw.value,
            H1_2_JointIndex.kRightElbowPitch.value,
        ]
        return motor_index.value in weak_motors
    
    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            H1_2_JointIndex.kLeftElbowRoll.value,
            H1_2_JointIndex.kLeftWristPitch.value,
            H1_2_JointIndex.kLeftWristyaw.value,
            H1_2_JointIndex.kRightElbowRoll.value,
            H1_2_JointIndex.kRightWristPitch.value,
            H1_2_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors
    
class H1_2_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 13
    kLeftShoulderRoll = 14
    kLeftShoulderYaw = 15
    kLeftElbowPitch = 16
    kLeftElbowRoll = 17
    kLeftWristPitch = 18
    kLeftWristyaw = 19

    # Right arm
    kRightShoulderPitch = 20
    kRightShoulderRoll = 21
    kRightShoulderYaw = 22
    kRightElbowPitch = 23
    kRightElbowRoll = 24
    kRightWristPitch = 25
    kRightWristYaw = 26

class H1_2_JointIndex(IntEnum):
    # Left leg
    kLeftHipYaw = 0
    kLeftHipRoll = 1
    kLeftHipPitch = 2
    kLeftKnee = 3
    kLeftAnkle = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipYaw = 6
    kRightHipRoll = 7
    kRightHipPitch = 8
    kRightKnee = 9
    kRightAnkle = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12

    # Left arm
    kLeftShoulderPitch = 13
    kLeftShoulderRoll = 14
    kLeftShoulderYaw = 15
    kLeftElbowPitch = 16
    kLeftElbowRoll = 17
    kLeftWristPitch = 18
    kLeftWristyaw = 19

    # Right arm
    kRightShoulderPitch = 20
    kRightShoulderRoll = 21
    kRightShoulderYaw = 22
    kRightElbowPitch = 23
    kRightElbowRoll = 24
    kRightWristPitch = 25
    kRightWristYaw = 26

    kNotUsedJoint0 = 27
    kNotUsedJoint1 = 28
    kNotUsedJoint2 = 29
    kNotUsedJoint3 = 30
    kNotUsedJoint4 = 31
    kNotUsedJoint5 = 32
    kNotUsedJoint6 = 33
    kNotUsedJoint7 = 34

class H1_ArmController:
    def __init__(self, simulation_mode = False):
        self.simulation_mode = simulation_mode
        
        logger_mp.info("Initialize H1_ArmController...")
        self.q_target = np.zeros(8)
        self.tauff_target = np.zeros(8)

        self.kp_high = 300.0
        self.kd_high = 5.0
        self.kp_low = 140.0
        self.kd_low = 3.0

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

        self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, go_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, go_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[H1_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[H1_ArmController] Subscribe dds ok.")

        # initialize h1's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_go_msg_dds__LowCmd_()
        self.msg.head[0] = 0xFE
        self.msg.head[1] = 0xEF
        self.msg.level_flag = 0xFF
        self.msg.gpio = 0

        self.all_motor_q = self.get_current_motor_q()
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")

        for id in H1_JointIndex:
            if self._Is_weak_motor(id):
                self.msg.motor_cmd[id].kp = self.kp_low
                self.msg.motor_cmd[id].kd = self.kd_low
                self.msg.motor_cmd[id].mode = 0x01
            else:
                self.msg.motor_cmd[id].kp = self.kp_high
                self.msg.motor_cmd[id].kd = self.kd_high
                self.msg.motor_cmd[id].mode = 0x0A
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]
        logger_mp.info("Lock OK!")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize H1_ArmController OK!")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = H1_LowState()
                for id in range(H1_Num_Motors):
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target     = self.q_target
                arm_tauff_target = self.tauff_target

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

            for idx, id in enumerate(H1_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]      

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # logger_mp.debug(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # logger_mp.debug(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target, tauff_target):
        '''Set control target values q & tau of the left and right arm motors.'''
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H1_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H1_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in H1_JointArmIndex])
    
    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
        logger_mp.info("[H1_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(8)
            # self.tauff_target = np.zeros(8)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                logger_mp.info("[H1_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            H1_JointIndex.kLeftAnkle.value,
            H1_JointIndex.kRightAnkle.value,
            # Left arm
            H1_JointIndex.kLeftShoulderPitch.value,
            H1_JointIndex.kLeftShoulderRoll.value,
            H1_JointIndex.kLeftShoulderYaw.value,
            H1_JointIndex.kLeftElbow.value,
            # Right arm
            H1_JointIndex.kRightShoulderPitch.value,
            H1_JointIndex.kRightShoulderRoll.value,
            H1_JointIndex.kRightShoulderYaw.value,
            H1_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors
    
class H1_JointArmIndex(IntEnum):
    # Unlike G1 and H1_2, the arm order in DDS messages for H1 is right then left. 
    # Therefore, the purpose of switching the order here is to maintain consistency with G1 and H1_2.
    # Left arm
    kLeftShoulderPitch = 16
    kLeftShoulderRoll = 17
    kLeftShoulderYaw = 18
    kLeftElbow = 19
    # Right arm
    kRightShoulderPitch = 12
    kRightShoulderRoll = 13
    kRightShoulderYaw = 14
    kRightElbow = 15

class H1_JointIndex(IntEnum):
    kRightHipRoll = 0
    kRightHipPitch = 1
    kRightKnee = 2
    kLeftHipRoll = 3
    kLeftHipPitch = 4
    kLeftKnee = 5
    kWaistYaw = 6
    kLeftHipYaw = 7
    kRightHipYaw = 8
    kNotUsedJoint = 9
    kLeftAnkle = 10
    kRightAnkle = 11
    # Right arm
    kRightShoulderPitch = 12
    kRightShoulderRoll = 13
    kRightShoulderYaw = 14
    kRightElbow = 15
    # Left arm
    kLeftShoulderPitch = 16
    kLeftShoulderRoll = 17
    kLeftShoulderYaw = 18
    kLeftElbow = 19

class H2_ArmController:
    def __init__(self, motion_mode=False, simulation_mode=False):
        logger_mp.info("Initialize H2_ArmController...")
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)
        self.motion_mode = motion_mode
        self.simulation_mode = simulation_mode
        self.kp_high = 300.0
        self.kd_high = 5.0
        self.kp_low = 140.0
        self.kd_low = 3.0
        self.kp_wrist = 50.0
        self.kd_wrist = 2.0

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None
        
        if self.motion_mode:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
        else:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[H2_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[H2_ArmController] Subscribe dds ok.")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        logger_mp.debug(f"Current all body motor state q:\n{self.all_motor_q} \n")
        logger_mp.debug(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except two arms...")

        arm_indices = set(member.value for member in H2_JointArmIndex)
        for id in H2_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            logger_mp.info(
                f"Motor {id.value} ({id.name}): kp={self.msg.motor_cmd[id].kp}, kd={self.msg.motor_cmd[id].kd}"
            )
            self.msg.motor_cmd[id].q = self.all_motor_q[id]
        logger_mp.info("Lock OK!")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize H2_ArmController OK!")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = H2_LowState()
                for id in range(35):
                    lowstate.motor_state[id].q = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        if self.motion_mode:
            self.msg.motor_cmd[H2_JointIndex.kNotUsedJoint0].q = 1.0

        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target = self.q_target
                arm_tauff_target = self.tauff_target

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit=self.arm_velocity_limit)

            for idx, id in enumerate(H2_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)

    def ctrl_dual_arm(self, q_target, tauff_target):
        """Set control target values q & tau of the left and right arm motors."""
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target

    def get_mode_machine(self):
        """Return current dds mode machine."""
        return self.lowstate_subscriber.Read().mode_machine

    def get_current_motor_q(self):
        """Return current state q of all body motors."""
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H2_JointIndex])

    def get_current_dual_arm_q(self):
        """Return current state q of the left and right arm motors."""
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in H2_JointArmIndex])

    def get_current_dual_arm_dq(self):
        """Return current state dq of the left and right arm motors."""
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in H2_JointArmIndex])

    def ctrl_dual_arm_go_home(self):
        """Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero."""
        logger_mp.info("[H2_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(14)
        tolerance = 0.05
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                if self.motion_mode:
                    for weight in np.linspace(1, 0, num=101):
                        self.msg.motor_cmd[H2_JointIndex.kNotUsedJoint0].q = weight
                        time.sleep(0.02)
                logger_mp.info("[H2_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t=5.0):
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            H2_JointIndex.kLeftAnklePitch.value,
            H2_JointIndex.kRightAnklePitch.value,
            # Left arm
            H2_JointIndex.kLeftShoulderPitch.value,
            H2_JointIndex.kLeftShoulderRoll.value,
            H2_JointIndex.kLeftShoulderYaw.value,
            H2_JointIndex.kLeftElbow.value,
            # Right arm
            H2_JointIndex.kRightShoulderPitch.value,
            H2_JointIndex.kRightShoulderRoll.value,
            H2_JointIndex.kRightShoulderYaw.value,
            H2_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors

    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            H2_JointIndex.kLeftWristRoll.value,
            H2_JointIndex.kLeftWristPitch.value,
            H2_JointIndex.kLeftWristyaw.value,
            H2_JointIndex.kRightWristRoll.value,
            H2_JointIndex.kRightWristPitch.value,
            H2_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors

class H2_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28


class H2_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

    # Head
    kHeadPitch = 29
    kHeadYaw = 30

    # not used
    kNotUsedJoint0 = 31
    kNotUsedJoint1 = 32
    kNotUsedJoint2 = 33
    kNotUsedJoint3 = 34

if __name__ == "__main__":
    from robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK, H2_ArmIK
    import pinocchio as pin

    ChannelFactoryInitialize(1) # 0 for real robot, 1 for simulation

    # arm_ik = G1_29_ArmIK(Unit_Test = True, Visualization = False)
    # arm = G1_29_ArmController(simulation_mode=True)
    # arm_ik = G1_23_ArmIK(Unit_Test = True, Visualization = False)
    # arm = G1_23_ArmController()
    # arm_ik = H1_2_ArmIK(Unit_Test = True, Visualization = False)
    # arm = H1_2_ArmController()
    # arm_ik = H1_ArmIK(Unit_Test = True, Visualization = True)
    # arm = H1_ArmController()
    arm_ik = H2_ArmIK(Unit_Test = True, Visualization = False)
    arm = H2_ArmController()


    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, +0.25, 0.1]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.25, 0.1]),
    )

    rotation_speed = 0.005  # Rotation speed in radians per iteration

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program): \n")
    if user_input.lower() == 's':
        step = 0
        arm.speed_gradual_max()
        while True:
            if step <= 120:
                angle = rotation_speed * step
                L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
                R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

                L_tf_target.translation += np.array([0.001,  0.001, 0.001])
                R_tf_target.translation += np.array([0.001, -0.001, 0.001])
            else:
                angle = rotation_speed * (240 - step)
                L_quat = pin.Quaternion(np.cos(angle / 2), 0, np.sin(angle / 2), 0)  # y axis
                R_quat = pin.Quaternion(np.cos(angle / 2), 0, 0, np.sin(angle / 2))  # z axis

                L_tf_target.translation -= np.array([0.001,  0.001, 0.001])
                R_tf_target.translation -= np.array([0.001, -0.001, 0.001])

            L_tf_target.rotation = L_quat.toRotationMatrix()
            R_tf_target.rotation = R_quat.toRotationMatrix()

            current_lr_arm_q  = arm.get_current_dual_arm_q()
            current_lr_arm_dq = arm.get_current_dual_arm_dq()

            sol_q, sol_tauff = arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous, current_lr_arm_q, current_lr_arm_dq)

            arm.ctrl_dual_arm(sol_q, sol_tauff)

            step += 1
            if step > 240:
                step = 0
            time.sleep(0.01)
