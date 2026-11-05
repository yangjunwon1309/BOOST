# Runtime protocols

## PATH arm command: `G1_PATH_UPPER_COMMAND`

Transport is LCM over the PATH multicast URL. A packet is exactly 192 bytes, little-endian, with packed layout `<IHHQq42f`:

| Offset | Field | Type |
|---:|---|---|
| 0 | magic `PDU1` (`0x31554450`) | `uint32` |
| 4 | version (`1`) | `uint16` |
| 6 | flags: bit 1 closes left, bit 2 closes right (MuJoCo compatibility adapter only) | `uint16` |
| 8 | sequence | `uint64` |
| 16 | PATH monotonic timestamp, microseconds | `int64` |
| 24 | 14 arm positions, radians | `float32[14]` |
| 80 | 14 arm velocities, radians/second | `float32[14]` |
| 136 | 14 feed-forward torques, Nm | `float32[14]` |

<<<<<<< HEAD
Arm order is G1 motor slots 15–28: left shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw, then the same seven right-arm joints. PDU1 contains no leg or waist targets. BOOST's G1 adapter maps the arm fields to `rt/arm_sdk`; DreamWaQ consumes that topic with `rt/lowstate` and merges its lower-body target before writing the sole hardware `rt/lowcmd` command.
=======
Arm array order is G1 motor order 15-28: left shoulder pitch/roll/yaw,
elbow, wrist roll/pitch/yaw, then the same seven right-arm joints. This packet
contains no leg, waist, or hand joint command. A set grip bit means close; a
clear bit means open. Only MuJoCo's optional binary Revo2 compatibility adapter
uses these flags. The physical G1 SDK bridge ignores them; real Revo2 targets
come from the continuous hand MPPI DDS publisher.

The G1 bridge accepts this packet size, magic, version, defined flags, and
finite joint values. A received-command timeout releases arm SDK ownership;
the bridge's default timeout is 150 ms.
>>>>>>> 228019f83417f222bc20e79b837865e645492e13

## G1 state: `G1_ROBOT_STATE`

Transport is LCM. The packet is exactly 461 bytes, little-endian, with packed layout `<IHHQQqB107f`. It contains GST1 magic/version, bridge and lowstate sequences, G1 system-clock timestamp, mode-machine byte, base/torso IMU data, and 29 each of measured position, velocity, and estimated torque. PATH uses base IMU and joint state for gravity estimation and IK seeds. The G1 bridge preserves this layout.

## Separate policy and camera packets

The target g1_ws `G1_POLICY_ACTION` GCA1 packet carries 29 policy values. It is not compatible with PDU1 and is not consumed as a PATH arm target. The target `G1_CAMERA_FRAME` CRAM packet is a separate camera stream.

Teleimager's recording stream uses TIM2 with a 44-byte `<4sHHQQQIII>` header: magic/version/flags, capture sequence, G1 wall and monotonic capture times, width, height, and JPEG length. The G1 preset uses head port 55558 and wrist ports 55559/55560. The Thor wrist relay validates TIM2 and publishes wrist-image message v2 with the JPEG, sequence, G1 capture timestamps, and Thor receive timestamp. Its receiver continues to accept v1 frames during transition.

G1 and Thor monotonic timestamps are from different clock domains; do not compare them directly. The Thor receive timestamp remains the freshness clock. Sequence numbers expose frames lost before or during relay. Teleimager RGB does not replace the separate D405 RGB-D capture and depth synchronization path.

<<<<<<< HEAD
## Hand DDS

The PATH bridge's optional Dex1 route uses Unitree `MotorCmds_` and `MotorStates_` on `rt/dex1/{left,right}/{cmd,state}`. The active DreamWaQ launcher disables that backend. Physical BrainCo Revo2 uses separate six-channel `MotorCmds_`/`MotorStates_` topics on `rt/brainco/{left,right}/{cmd,state}`; BOOST currently runs this route in MuJoCo and leaves physical hand-service deployment separate.
=======
The current local HOMIE merge forwards arm position, gains, and feed-forward
torque but sends zero target velocity on `rt/lowcmd`; PATH's intermediate
`dq` field is not consumed by that merge.

## BrainCo Revo2 service and hand-policy transport

Revo2 uses Unitree's external [`brainco_hand_service`](https://github.com/unitreerobotics/brainco_hand_service) over
`rt/brainco/{left,right}/cmd` and `rt/brainco/{left,right}/state`; each hand
has six normalized values in `[0, 1]`, ordered Thumb, Thumb_aux, Index,
Middle, Ring, Pinky. `boost/control/hand_mppi/` defines the shared mapping and
`BrainCoDDS` transport. On a physical G1, Thor publishes fresh retargeted
targets directly to those DDS topics and receives the measured state. The G1
service owns the hand serial links; the arm bridge does not publish hand DDS.
The external service is not vendored. HOMIE continues to own `rt/lowcmd`.
>>>>>>> 228019f83417f222bc20e79b837865e645492e13
