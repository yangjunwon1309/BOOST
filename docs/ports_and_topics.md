# Ports and topics

These are the PATH-derived deployment defaults currently carried by BOOST.
LCM uses UDP multicast; Unitree DDS
topics are local SDK channels and do not use the ZMQ TCP ports below.

| Port / transport | Publisher -> consumer | Topic / channel | Purpose |
| --- | --- | --- | --- |
| ZMQ TCP 5565 | PICO or Galaxy XR -> PATH state machine | `pico_button_event`, `pico_controller_state` | Arm tracking and Y/B belief selection |
| ZMQ TCP 5566 | PICO or Galaxy XR -> PATH state machine | `vr_pose_debug` | Calibrated absolute wrist poses |
| ZMQ TCP 5560 | PATH state machine -> viewers | `path_qp_reference` | Arm IK and belief telemetry |
| ZMQ TCP 5563 | Operator tool -> PATH state machine | `path_state_command` | State and target controls |
| ZMQ TCP 5564 | Perception -> PATH state machine | `path_static_ee_target` | Selected surface/contact target |
| ZMQ TCP 5567 | PATH state machine -> belief gateway | `path_shared_control_hand` | Selected hand and freshness |
| HTTP TCP 8080 | Perception web server -> operator browser | HTTP | Mask, geometry, and belief view |
| HTTP TCP 8082 | Remote wrist viewer -> operator browser | HTTP | Optional SAM3 wrist-mask and target view |
| LCM UDP 7667, group `239.255.76.67` | Thor PATH <-> G1 bridge | `G1_PATH_UPPER_COMMAND`, `G1_ROBOT_STATE` | Arm targets and measured robot state |
| LCM UDP 7667, group `239.255.77.67` | Thor <-> Wi-Fi viewer/remote worker | `PATH_V11_WRIST_*` | Remote SAM3 camera/result and visualization |
| Unitree DDS | G1 -> bridge | `rt/lowstate` | Measured G1 state |
| Unitree DDS | G1 bridge -> local HOMIE | `rt/arm_sdk` | PATH arm targets and SDK gains/torque |
| Unitree DDS | Local HOMIE -> G1 | `rt/lowcmd` | Final combined 29-DoF command |
| Unitree DDS | G1 bridge -> Unitree `dex1_1_service` | `rt/dex1/{left,right}/cmd` (`MotorCmds_`) | One position command per hand |
| Unitree DDS | Unitree `dex1_1_service` -> G1 bridge | `rt/dex1/{left,right}/state` (`MotorStates_`) | Measured hand state and response monitoring |

TCP ports 5556-5559 and the `G1_PATH_MOTOR_COMMAND` full-body LCM packet are
not used by this deployment. Do not allocate them to PATH processes.
The G1 bridge never publishes `rt/lowcmd`; local HOMIE owns that final
whole-body topic. HOMIE settings are owned by its separate deployment and must
preserve the joint ownership in [`control_ownership.md`](control_ownership.md).
