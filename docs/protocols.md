# Runtime protocols

## PATH arm command: G1_PATH_UPPER_COMMAND

Transport is LCM over the PATH multicast URL. PDU1 is exactly 192 bytes,
little-endian, with packed layout <code>&lt;IHHQq42f</code>:

| Offset | Field | Type |
| ---: | --- | --- |
| 0 | Magic PDU1 (0x31554450) | uint32 |
| 4 | Version (1) | uint16 |
| 6 | Flags: bit 1 closes left, bit 2 closes right; MuJoCo compatibility only | uint16 |
| 8 | Sequence | uint64 |
| 16 | PATH monotonic timestamp, microseconds | int64 |
| 24 | 14 arm positions, radians | float32[14] |
| 80 | 14 arm velocities, radians/second | float32[14] |
| 136 | 14 feed-forward torques, Nm | float32[14] |

Arm order is G1 motor slots 15-28: left shoulder pitch/roll/yaw, elbow, wrist
roll/pitch/yaw, then the same seven right-arm joints. PDU1 contains no leg,
waist, or continuous hand targets. The G1 adapter maps its arm fields to
rt/arm_sdk; DreamWaQ merges those targets with its body policy and owns the
sole physical rt/lowcmd publisher. The bridge validates size, magic, version,
flags, and finite joint values. Its command timeout releases arm-SDK ownership.

## G1 state: G1_ROBOT_STATE

Transport is LCM. GST1 is exactly 461 bytes, little-endian, with packed layout
<code>&lt;IHHQQqB107f</code>. It contains magic/version, bridge and lowstate sequences, a G1
system-clock timestamp, mode-machine byte, base/torso IMU data, and 29 each of
measured position, velocity, and estimated torque. The adapter preserves this
layout for PATH state and IK feedback.

## Separate policy and camera packets

The target g1_ws G1_POLICY_ACTION GCA1 packet carries 29 policy values. It
is not compatible with PDU1 and does not replace PATH arm commands. The target
G1_CAMERA_FRAME CRAM packet is a separate camera stream.

Teleimager's recording stream uses TIM2 with a 44-byte <code>&lt;4sHHQQQIII&gt;</code> header:
magic/version/flags, capture sequence, G1 wall and monotonic capture times,
width, height, and JPEG length. The G1 preset uses head port 55558 and wrist
ports 55559/55560. Thor validates TIM2 and publishes wrist-image message v2
with JPEG, sequence, G1 capture timestamps, and Thor receive timestamp; its
receiver also accepts v1 frames during transition.

G1 and Thor monotonic timestamps belong to different clock domains and must
not be compared directly. Thor's receive timestamp is the freshness clock.
Sequence numbers expose frames lost before or during relay. Teleimager RGB
does not replace the D405 RGB-D capture and depth synchronization path.

## BrainCo Revo2 DDS

The external
[brainco_hand_service](https://github.com/unitreerobotics/brainco_hand_service)
owns the Revo2 serial links. Unitree DDS topics are
rt/brainco/{left,right}/cmd (MotorCmds_) and
rt/brainco/{left,right}/state (MotorStates_). Each hand has six normalized
values in [0, 1], ordered Thumb, Thumb_aux, Index, Middle, Ring, Pinky.
Thor's hand MPPI process publishes fresh targets and reads measured state; the
G1 arm bridge does not publish hand commands. Revo2 hands are independent of
DreamWaQ's 29-joint rt/lowcmd body array.
