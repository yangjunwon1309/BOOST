# Ports and topics

These are the active BOOST hand-tracking teleoperation, simulation, and camera
interfaces. LCM uses UDP multicast. Unitree DDS topics are separate from the
TCP endpoints in this table. The default wired link is Thor `192.168.123.2` to
G1 `192.168.123.164` on `eth0`; DreamWaQ's config is authoritative if those
addresses have been changed. Galaxy XR reaches Thor through its headset-facing
LAN address.

| Port / transport | Publisher -> consumer | Topic / channel | Purpose |
| --- | --- | --- | --- |
| HTTPS/WSS TCP 8012 | Galaxy XR headset -> Thor | Televuer | Secure WebXR session and hand tracking |
| ZMQ TCP 5566 | Galaxy XR / Televuer -> PATH and hand MPPI | vr_pose_debug | Hand-tracked wrist poses and landmarks |
| ZMQ TCP 5560 | PATH -> viewers | path_qp_reference | Arm IK and reference telemetry |
| ZMQ TCP 5563 | Operator tool -> PATH | path_state_command | State and target controls |
| ZMQ TCP 5564 | Perception -> PATH | path_static_ee_target | Selected surface/contact target |
| ZMQ TCP 5567 | PATH -> belief gateway | path_shared_control_hand | Selected hand and freshness |
| HTTP TCP 8080 | Perception server -> browser | HTTP | Mask, geometry, and operator view |
| HTTP TCP 8082 | Remote wrist viewer -> browser | HTTP | Optional wrist-mask and target view |
| LCM UDP 7667, 239.255.76.67 | Thor PATH <-> G1 adapter | G1_PATH_UPPER_COMMAND, G1_ROBOT_STATE | PDU1 arm targets and GST1 robot state |
| LCM UDP 7667, 239.255.77.67 | G1 camera / Thor relay <-> worker | PATH_V11_WRIST_* | TIM2-derived wrist RGB and perception |
| ZMQ TCP 55555 | Thor head Teleimager -> local clients | TIM2 preview | Head stereo preview |
| ZMQ TCP 55556-55557 | Disabled by current presets | Legacy preview ports | Not used by the current recording path |
| ZMQ TCP 55558 | Thor head Teleimager -> PATH experiment recorder | TIM2 recording | Timestamped stereo head JPEG |
| ZMQ TCP 55559-55560 | G1 Teleimager -> Thor recorder/relay | TIM2 recording | Timestamped left/right wrist JPEG |
| ZMQ TCP 5571, loopback | Wrist surface gateway <-> hand MPPI | Local query protocol | Calibrated point-cloud surface query; opt-in |
| UDP 15001 | Thor -> G1 DreamWaQ | DreamWaQ remote velocity command | Base velocity; configured bind/allowed-host addresses are in the DreamWaQ G1 config |
| ZMQ TCP 60000 | Clients -> each Teleimager host | Camera configuration request | Camera endpoint discovery, per host |
| TCP 60001-60003 | Galaxy XR / clients -> Teleimager host | WebRTC signaling and media | Used only when WebRTC camera transport is enabled |
| Unitree DDS | G1 -> DreamWaQ and adapter | rt/lowstate | Measured body state |
| Unitree DDS | G1 adapter -> DreamWaQ | rt/arm_sdk | PATH arm targets, gains, and feed-forward torque |
| Unitree DDS | DreamWaQ -> G1 | rt/lowcmd | Sole merged hardware body command |
| Unitree DDS, loopback in sim | PATH bridge -> MuJoCo | rt/arm_sdk | Simulated PATH arm targets |
| Unitree DDS, loopback in sim | MuJoCo -> PATH bridge | rt/lowstate | Simulated G1 state |
| Unitree DDS | Thor hand MPPI -> BrainCo service | rt/brainco/{left,right}/cmd | Six normalized Revo2 targets per hand |
| Unitree DDS | BrainCo service -> Thor hand MPPI | rt/brainco/{left,right}/state | Six measured Revo2 values per hand |

TCP 5565 is unassigned in the current hand teleop stack. Terminal controls in
the PATH process are `r` (start/resume), `s`
(recording start/stop), and `q` (stop and exit). DDS uses domain 0 on the wired
interface; its participant ports are assigned dynamically, so inspect them
with `ss -lunp` rather than assuming a fixed port.

The PATH experiment recorder reads the head preset from
external/teleimager/config/dex1_cam_nature_head_stereo.yaml and the G1 wrist
preset from external/teleimager/config/hbvcam_4m2214hd_g1.yaml. Host overrides
and relay settings are in configs/teleop/teleimager_relay.yaml and
configs/teleop/local.env.
See [protocols.md](protocols.md) for packet layouts and ownership details.
