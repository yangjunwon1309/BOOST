# PATH teleop source map

The PATH-derived teleoperation code in this directory starts from
`yangjunwon1309/PATH-G1` commit
`f98f8854cb45cc60afe1565f0eee6912d62aba64` (`homie-g1-snapshot`). BOOST owns
the relocated copy and may change it independently of PATH.

| PATH-G1 source | BOOST location | Role |
|---|---|---|
| `path_core/` | `boost/teleop/path_core/` | Shared state, references, and state-machine types |
| `path_core/intention/` | `boost/teleop/legacy/intention/` | Legacy intent/reference inference retained for the current bridge |
| `path_core/teleop/` | `boost/teleop/xr/` | XR gesture and arm-command helpers |
| `path_core/messaging/` | `boost/io/path_transport/` | PATH message schema, LCM protocol, and ZMQ transport |
| `path_streaming/` | `boost/teleop/path_streaming/` | PICO/Galaxy XR input and PATH state-machine bridge |
| `path_planning/` | `boost/control/arm/path_planning/` | G1 arm IK |
| `g1_bridge/` | `boost/io/g1_bridge/` | G1 arm SDK bridge; hand commands are Revo2 DDS |
| `perception/` | `services/pose_service/` | Wrist RGB-D, SAM3, geometry, and existing target publisher |
| `path_deploy/scripts/` | `scripts/deploy/` | PATH direct-LCM deployment shim |

The legacy intention package remains isolated because the existing PATH state
machine imports it. Removing that behavior is a planned BOOST teleop change.
<<<<<<< HEAD
The carried G1 bridge only understands the PATH arm packet and Dex1 flags; it
does not implement BrainCo Revo2 commands. The target DreamWaQ runtime is an
external component and owns hardware `rt/lowcmd`.
=======
The G1 bridge publishes PATH arm targets on `rt/arm_sdk`. Thor's hand MPPI
publishes BrainCo Revo2 targets directly over Unitree DDS. The external
BrainCo serial service owns the physical hands, while HOMIE/WBC remains the
sole publisher to `rt/lowcmd`.
>>>>>>> 228019f83417f222bc20e79b837865e645492e13
