# Ports and topics

These are the active BOOST teleop, sim, and camera defaults. LCM uses UDP multicast; Unitree DDS topics are independent of the TCP endpoints below.

| Port / transport | Publisher → consumer | Topic / channel | Purpose |
|---|---|---|---|
| ZMQ TCP 5565 | Galaxy XR / Televuer → PATH state machine | `pico_button_event`, `pico_controller_state` | Buttons and teleop state |
| ZMQ TCP 5566 | Galaxy XR / Televuer → PATH state machine | `vr_pose_debug` | Wrist poses and hand landmarks |
| ZMQ TCP 5560 | PATH state machine → viewers | `path_qp_reference` | Arm IK and reference telemetry |
| ZMQ TCP 5563–5567 | PATH/perception/control | configured in `configs/teleop/zmq_topics.yaml` | State, end-effector, and hand-selection control |
| HTTP TCP 8080/8082 | Perception server → browser | HTTP | Mask, geometry, and wrist view |
| LCM UDP 7667, `239.255.76.67` | Thor PATH ↔ G1 adapter | `G1_PATH_UPPER_COMMAND`, `G1_ROBOT_STATE` | PDU1 arm command and GST1 robot state |
| LCM UDP 7667, `239.255.77.67` | G1 camera → Thor/worker | `PATH_V11_WRIST_RGB_FRAME`, `PATH_V11_WRIST_*` | TIM2-derived wrist RGB and perception |
| ZMQ TCP 55558 | G1 Teleimager → camera clients | TIM2 head stream | Head camera RGB |
| ZMQ TCP 55559 | G1 Teleimager → Thor relay | TIM2 left-wrist stream | Left wrist RGB |
| ZMQ TCP 55560 | G1 Teleimager → Thor relay | TIM2 right-wrist stream | Right wrist RGB |
| Unitree DDS | G1 → DreamWaQ / adapter | `rt/lowstate` | Measured body state |
| Unitree DDS | G1 adapter → DreamWaQ | `rt/arm_sdk` | PATH arm targets and SDK gains/torque |
| Unitree DDS | DreamWaQ → G1 | `rt/lowcmd` | Sole merged hardware body command |
| Unitree DDS, loopback in sim | PATH sim bridge → MuJoCo | `rt/arm_sdk` | Simulated PATH arm targets |
| Unitree DDS, loopback in sim | MuJoCo → PATH sim bridge | `rt/lowstate` | Simulated G1 state |
| Unitree DDS | Revo2 sim / external service | `rt/brainco/{left,right}/{cmd,state}` | Six-channel hand targets/state |

The Teleimager wrist relay reads its ports and 640×480 dimensions from the selected server preset in `external/teleimager/config/hbvcam_4m2214hd_g1.yaml`. The preset and relay config can be overridden in `configs/teleop/local.env`.

The target `G1_POLICY_ACTION` GCA1 packet is a separate 29-joint format; PATH remains on PDU1. The G1 adapter never publishes `rt/lowcmd`. Do not run the target g1_ws low-level bridge with DreamWaQ because it creates a competing lowcmd writer. D405 RGB-D and its calibration/synchronization topics remain separate from TIM2 JPEG.
