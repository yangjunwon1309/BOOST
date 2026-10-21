# External runtime components

BOOST owns arm teleoperation, perception integration, the packaged robot
models, and the G1-local arm/Dex1 DDS bridge carried over from PATH. These
runtime components are installed separately:

- **Unitree SDK2** builds and runs `boost/io/g1_bridge/`. The local MuJoCo
  bring-up uses this DDS bridge on loopback, with the simulator acting as the
  low-state endpoint.
- **Local HOMIE/WBC** owns legs and waist, merges PATH's `rt/arm_sdk` arm data,
  and remains the only `rt/lowcmd` publisher. Its source snapshot is included
  at [`homie_g1_29dof/`](homie_g1_29dof/) without upstream Git metadata; its
  source and scoped license are recorded there. See
  [`../docs/homie_integration.md`](../docs/homie_integration.md).
- **Unitree `dex1_1_service`** owns the Dex1-1 serial connection. It publishes
  hand state and consumes the DDS command topics from `boost/io/g1_bridge/`; see
  [`../docs/g1_installation.md`](../docs/g1_installation.md).
- **XRoboToolkit / PICO SDK** supplies the live PICO manager's controller input.
- **Televuer and Unitree SDK2 Python** are installed separately under
  `external/televuer`, `external/unitree_sdk2_python`, and
  `external/unitree_sdk2`; `bash scripts/setup_external_runtime.sh` prepares
  these checkouts and the `.venv_runtime` environment.
- **Remote SAM3 worker** runs on the selected inference host and sends camera
  segmentation and geometry to the Thor gateway.

The checked-in hardware bridge still sends Dex1 open/close commands. The PATH
Sonic MuJoCo adapter and model are included for Revo2 simulation, with PATH grip
flags translated to the BrainCo DDS command topics; this does not add Revo2
serial/service support for physical hardware. Other external runtime projects
are not vendored here.
Keep host-specific checkout paths and credentials in deployment environment
files, not BOOST source. The HOMIE snapshot has no upstream commit metadata;
record an archive checksum when producing a deployment package.
