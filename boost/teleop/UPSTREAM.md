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
| `g1_bridge/` | `boost/io/g1_bridge/` | G1 arm SDK and PATH-era Dex1 bridge |
| `perception/` | `services/pose_service/` | Wrist RGB-D, SAM3, geometry, and existing target publisher |
| `path_deploy/scripts/` | `scripts/deploy/` | PATH direct-LCM deployment shim |

The legacy intention package remains isolated because the existing PATH state
machine imports it. Removing that behavior is a planned BOOST teleop change.
The carried G1 bridge only understands the PATH arm packet and Dex1 flags; it
does not implement BrainCo Revo2 commands. HOMIE/WBC remains an external
component and continues to own `rt/lowcmd`.
