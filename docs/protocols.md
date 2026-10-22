# Runtime protocols

## PATH arm command: `G1_PATH_UPPER_COMMAND`

Transport is LCM over the PATH multicast URL. A packet is exactly 192 bytes,
little-endian, with packed layout `<IHHQq42f`:

| Offset | Field | Type |
| ---: | --- | --- |
| 0 | magic `PDU1` (`0x31554450`) | `uint32` |
| 4 | version (`1`) | `uint16` |
| 6 | flags: bit 1 closes left Dex1, bit 2 closes right Dex1 | `uint16` |
| 8 | sequence | `uint64` |
| 16 | PATH monotonic timestamp, microseconds | `int64` |
| 24 | 14 arm positions, radians | `float32[14]` |
| 80 | 14 arm velocities, radians/second | `float32[14]` |
| 136 | 14 feed-forward torques, Nm | `float32[14]` |

Arm array order is G1 motor order 15-28: left shoulder pitch/roll/yaw,
elbow, wrist roll/pitch/yaw, then the same seven right-arm joints. This packet
contains no leg or waist command. A set grip bit means close; a clear bit means
open. The G1 bridge routes changes to the corresponding Dex1 hand DDS topic.
Grip commands are ignored in dry-run mode. Hand targets remain latched until a
later packet changes the bit; bridge shutdown commands both hands open.

The G1 bridge accepts this packet size, magic, version, defined flags, and
finite joint values. A received-command timeout releases arm SDK ownership;
the ported bridge's default timeout is 150 ms. The timeout does not change the
latched hand target.

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
ported `g1_ws` `kps`/`kds` arrays in `boost/io/g1_bridge/include/robot_parameters.hpp`;
PATH supplies `q`, `dq`, and feed-forward `tau`. The local HOMIE
deployment subscribes to `rt/arm_sdk`, merges the arms with WBC output, and
publishes the final command on `rt/lowcmd`. Topic and marker usage
follow Unitree's [G1 arm SDK DDS example](https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/g1/high_level/g1_arm7_sdk_dds_example.cpp).
The bridge fills other fixed-width motor entries from measured state; HOMIE
uses only the arm range and enable marker from this topic.

## Dex1-1 service DDS

The hand interface matches Unitree's [`dex1_1_service`](https://github.com/unitreerobotics/dex1_1_service):
the service subscribes to `rt/dex1/left/cmd` and `rt/dex1/right/cmd`, and
publishes `rt/dex1/left/state` and `rt/dex1/right/state`. Both sides use
Unitree SDK2's `unitree_go::msg::dds_::MotorCmds_` and `MotorStates_` DDS types.
Each topic carries one motor entry; motor ID 0 is right and ID 1 is left in the
service's serial mapping.

The PATH bridge subscribes to both state topics and publishes both command
topics. Each command sets `mode=1`, position `q` in radians, zero `dq` and
`tau`, `kp=5.0`, `kd=0.05`, and a zeroed reserve field, matching Unitree's
official Dex1-1 test publisher. The bridge sends at 200 Hz with bounded
position steps. Both `dex1_1_service` and `boost/io/g1_bridge` initialize Unitree DDS on
domain 0; their selected network interfaces must share DDS traffic. Run the
service on the computer connected to the grippers' serial cables. The bridge
does not open the serial ports itself.

These topics are separate from the body command path and do not change
`rt/lowcmd` ownership. The service publishes state; the PATH bridge publishes
commands. Grip flags from PATH select the targets sent to that service.

The current local HOMIE merge forwards arm position, gains, and feed-forward
torque but sends zero target velocity on `rt/lowcmd`; PATH's intermediate
`dq` field is not consumed by that merge.

## Revo2 hand service and hand-policy transport

The PATH bridge's Dex1 topics do not control BrainCo Revo2. Revo2 uses
Unitree's external `brainco_hand_service` over
`rt/brainco/{left,right}/cmd` and `rt/brainco/{left,right}/state`; each hand
has six normalized values in `[0, 1]`, ordered Thumb, Thumb_aux, Index,
Middle, Ring, Pinky. `boost/control/hand_mppi/` defines the shared Revo2 mapping
and `BrainCoDDS` transport; the simulator and future hand policy use the same
topics and IDL types. The external hand service is not vendored, and HOMIE
continues to own `rt/lowcmd`. PATH's LCM arm packet carries only binary grip
flags, so continuous hand targets travel directly over the BrainCo DDS topics.
