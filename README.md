# PATH-G1 deploy

PATH-G1 runs arm-only, perception-assisted teleoperation for Unitree G1.
Thor computes arm references with nonlinear IK and can blend SAM3 surface and
belief inputs. The G1 bridge publishes PATH commands only through Unitree's
arm SDK topic. HOMIE/WBC and the Dex1 hand service are separate owners.

## Repository map

- `path_streaming/`: PICO inputs, PATH state machine, arm references, and telemetry.
- `path_planning/`: nonlinear G1 arm IK.
- `perception/`: RGB-D capture, SAM3 masks, surface geometry, belief assistance,
  and end-effector target publication.
- `assets/robot/`: packaged G1 URDF, meshes, and upstream license.
- `g1_bridge/`: G1-local LCM/Unitree SDK arm bridge.
- `docs/`: control ownership, protocol layout, ports, and deployment order.

## Deployment sequence

1. Build and start `g1_bridge` on G1 in default dry-run mode; confirm
   `G1_ROBOT_STATE` reaches Thor.
2. Start the PATH state machine and the desired PICO input process on Thor.
3. Start the selected perception route. For wrist D405, run the Thor gateway
   and the remote SAM3 worker; the local web server is an optional camera view.
4. Start the local `homie_g1_29dof` deployment on G1 with
   `arm_source: "arm_sdk"` and its sole `rt/lowcmd` writer.
5. Restart the G1 bridge with `G1_BRIDGE_LIVE=1` to forward PATH arm commands
   on `rt/arm_sdk`; HOMIE merges them with its leg/waist output.

The state machine and IK default to `assets/robot/g1_body29_hand14.urdf`.
The G1 bridge build and launch instructions are in [`g1_bridge/README.md`](g1_bridge/README.md).

## Deployment contracts

- [Joint and process ownership](docs/control_ownership.md)
- [LCM and SDK packet formats](docs/protocols.md)
- [Ports and topics](docs/ports_and_topics.md)
- [Deployment order](docs/deployment.md)
- [HOMIE interface comparison](docs/homie_integration.md)
- [Perception and remote SAM3](perception/README.md)

HOMIE/WBC source is not vendored here. Select and pin that deployment in the
HOMIE repository; PATH-G1's actuator boundary is documented above.
