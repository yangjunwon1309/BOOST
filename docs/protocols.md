# Runtime protocols

## PATH arm command: `G1_PATH_UPPER_COMMAND`

Transport is LCM over the PATH multicast URL. A packet is exactly 192 bytes, little-endian, with packed layout `<IHHQq42f`:

| Offset | Field | Type |
|---:|---|---|
| 0 | magic `PDU1` (`0x31554450`) | `uint32` |
| 4 | version (`1`) | `uint16` |
| 6 | flags: bit 1 closes left Dex1, bit 2 closes right Dex1 | `uint16` |
| 8 | sequence | `uint64` |
| 16 | PATH monotonic timestamp, microseconds | `int64` |
| 24 | 14 arm positions, radians | `float32[14]` |
| 80 | 14 arm velocities, radians/second | `float32[14]` |
| 136 | 14 feed-forward torques, Nm | `float32[14]` |

Arm order is G1 motor slots 15–28: left shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw, then the same seven right-arm joints. PDU1 contains no leg or waist targets. BOOST's G1 adapter maps the arm fields to `rt/arm_sdk`; DreamWaQ consumes that topic with `rt/lowstate` and merges its lower-body target before writing the sole hardware `rt/lowcmd` command.

## G1 state: `G1_ROBOT_STATE`

Transport is LCM. The packet is exactly 461 bytes, little-endian, with packed layout `<IHHQQqB107f`. It contains GST1 magic/version, bridge and lowstate sequences, G1 system-clock timestamp, mode-machine byte, base/torso IMU data, and 29 each of measured position, velocity, and estimated torque. PATH uses base IMU and joint state for gravity estimation and IK seeds. The G1 bridge preserves this layout.

## Separate policy and camera packets

The target g1_ws `G1_POLICY_ACTION` GCA1 packet carries 29 policy values. It is not compatible with PDU1 and is not consumed as a PATH arm target. The target `G1_CAMERA_FRAME` CRAM packet is a separate camera stream.

Teleimager's recording stream uses TIM2 with a 44-byte `<4sHHQQQIII>` header: magic/version/flags, capture sequence, G1 wall and monotonic capture times, width, height, and JPEG length. The G1 preset uses head port 55558 and wrist ports 55559/55560. The Thor wrist relay validates TIM2 and publishes wrist-image message v2 with the JPEG, sequence, G1 capture timestamps, and Thor receive timestamp. Its receiver continues to accept v1 frames during transition.

G1 and Thor monotonic timestamps are from different clock domains; do not compare them directly. The Thor receive timestamp remains the freshness clock. Sequence numbers expose frames lost before or during relay. Teleimager RGB does not replace the separate D405 RGB-D capture and depth synchronization path.

## Hand DDS

The PATH bridge's optional Dex1 route uses Unitree `MotorCmds_` and `MotorStates_` on `rt/dex1/{left,right}/{cmd,state}`. The active DreamWaQ launcher disables that backend. Physical BrainCo Revo2 uses separate six-channel `MotorCmds_`/`MotorStates_` topics on `rt/brainco/{left,right}/{cmd,state}`; BOOST currently runs this route in MuJoCo and leaves physical hand-service deployment separate.
