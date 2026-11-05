# Ports and topics

These are the active BOOST teleoperation, simulation, camera, and controller
interfaces. LCM uses UDP multicast. Unitree DDS topics are separate from the
TCP endpoints in this table.

| Port / transport | Publisher -> consumer | Topic / channel | Purpose |
| --- | --- | --- | --- |
| ZMQ TCP 5565 | Galaxy XR / Televuer -> PATH | pico_button_event, pico_controller_state | Controller buttons and teleop state |
| ZMQ TCP 5566 | Galaxy XR / Televuer -> PATH and hand MPPI | vr_pose_debug | Wrist poses and hand landmarks |
| ZMQ TCP 5560 | PATH -> viewers | path_qp_reference | Arm IK and reference telemetry |
| ZMQ TCP 5563 | Operator tool -> PATH | path_state_command | State and target controls |
| ZMQ TCP 5564 | Perception -> PATH | path_static_ee_target | Selected surface/contact target |
| ZMQ TCP 5567 | PATH -> belief gateway | path_shared_control_hand | Selected hand and freshness |
| HTTP TCP 8080 | Perception server -> browser | HTTP | Mask, geometry, and operator view |
| HTTP TCP 8082 | Remote wrist viewer -> browser | HTTP | Optional wrist-mask and target view |
| LCM UDP 7667, 239.255.76.67 | Thor PATH <-> G1 adapter | G1_PATH_UPPER_COMMAND, G1_ROBOT_STATE | PDU1 arm targets and GST1 robot state |
| LCM UDP 7667, 239.255.77.67 | G1 camera / Thor relay <-> worker | PATH_V11_WRIST_* | TIM2-derived wrist RGB and perception |
| ZMQ TCP 55558 | G1 Teleimager -> camera clients | TIM2 head stream | Head RGB |
| ZMQ TCP 55559 | G1 Teleimager -> Thor relay | TIM2 left-wrist stream | Left wrist RGB |
| ZMQ TCP 55560 | G1 Teleimager -> Thor relay | TIM2 right-wrist stream | Right wrist RGB |
| ZMQ TCP 5571, loopback | Wrist surface gateway <-> hand MPPI | Local query protocol | Calibrated point-cloud surface query; opt-in |
| Unitree DDS | G1 -> DreamWaQ and adapter | rt/lowstate | Measured body state |
| Unitree DDS | G1 adapter -> DreamWaQ | rt/arm_sdk | PATH arm targets, gains, and feed-forward torque |
| Unitree DDS | DreamWaQ -> G1 | rt/lowcmd | Sole merged hardware body command |
| Unitree DDS, loopback in sim | PATH bridge -> MuJoCo | rt/arm_sdk | Simulated PATH arm targets |
| Unitree DDS, loopback in sim | MuJoCo -> PATH bridge | rt/lowstate | Simulated G1 state |
| Unitree DDS | Thor hand MPPI -> BrainCo service | rt/brainco/{left,right}/cmd | Six normalized Revo2 targets per hand |
| Unitree DDS | BrainCo service -> Thor hand MPPI | rt/brainco/{left,right}/state | Six measured Revo2 values per hand |

Teleimager wrist relay ports and image dimensions come from the selected
server preset in external/teleimager/config/hbvcam_4m2214hd_g1.yaml. The
preset and relay settings can be overridden in configs/teleop/local.env.
See [protocols.md](protocols.md) for packet layouts and ownership details.
