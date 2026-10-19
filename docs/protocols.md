# Runtime protocols

## PATH arm command: `G1_PATH_UPPER_COMMAND`

Transport is LCM over the PATH multicast URL. A packet is exactly 192 bytes,
little-endian, with packed layout `<IHHQq42f`:

| Offset | Field | Type |
| ---: | --- | --- |
| 0 | magic `PDU1` (`0x31554450`) | `uint32` |
| 4 | version (`1`) | `uint16` |
| 6 | flags: left/right grip intent in bits 1/2 | `uint16` |
| 8 | sequence | `uint64` |
| 16 | PATH monotonic timestamp, microseconds | `int64` |
| 24 | 14 arm positions, radians | `float32[14]` |
| 80 | 14 arm velocities, radians/second | `float32[14]` |
| 136 | 14 feed-forward torques, Nm | `float32[14]` |

Arm array order is G1 motor order 15-28: left shoulder pitch/roll/yaw,
elbow, wrist roll/pitch/yaw, then the same seven right-arm joints. This packet
contains no leg or waist command. Grip bits are metadata for the separate
Dex1 service and are not sent to Unitree DDS.

The G1 bridge accepts this packet size, magic, version, defined flags, and
finite joint values. A received-command timeout releases arm SDK ownership;
the ported bridge's default timeout is 150 ms. Grip flags are accepted as
packet metadata but do not publish hand commands.

## G1 state: `G1_ROBOT_STATE`

Transport is LCM. The state packet is exactly 461 bytes, little-endian, with
packed layout `<IHHQQqB107f`:

| Offset | Field |
| ---: | --- |
| 0 | magic `GST1` (`0x31545347`), version `1`, flags |
| 8 | bridge state sequence, lowstate sequence (`uint64` each) |
| 24 | G1 system-clock timestamp, microseconds (`int64`) |
| 32 | Unitree mode-machine byte |
| 33 | base quaternion, gyro, acceleration; torso quaternion, gyro, acceleration (20 floats) |
| 113 | G1 positions, velocities, estimated torques (29 floats each) |

The torso-IMU flag is set when the bridge receives `rt/secondary_imu`; if that
topic is unavailable, the flag stays clear and the torso fields remain zero.
PATH uses the base IMU and joint state for gravity estimation and arm IK seeds.

## Unitree SDK boundary

The bridge subscribes to `rt/lowstate` and publishes PATH arm targets through
`rt/arm_sdk`. It writes arm motor slots 15-28 and the reserved SDK enable
marker in slot 29; it does not publish `rt/lowcmd`. The arm gains come from the
ported `g1_ws` `kps`/`kds` arrays in `g1_bridge/include/robot_parameters.hpp`;
PATH supplies `q`, `dq`, and feed-forward `tau`. The local HOMIE
deployment subscribes to `rt/arm_sdk`, merges the arms with WBC output, and
publishes the final command on `rt/lowcmd`. Topic and marker usage
follow Unitree's [G1 arm SDK DDS example](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/g1/high_level/g1_arm7_sdk_dds_example.cpp).
The bridge fills other fixed-width motor entries from measured state; HOMIE
uses only the arm range and enable marker from this topic.

The current local HOMIE merge forwards arm position, gains, and feed-forward
torque but sends zero target velocity on `rt/lowcmd`; PATH's intermediate
`dq` field is not consumed by that merge.
