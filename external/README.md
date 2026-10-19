# External runtime components

PATH-G1 owns arm teleoperation, perception integration, the packaged G1 model,
and the G1-local arm/Dex1 DDS bridge. These runtime components are installed
separately:

- **Unitree SDK2** builds and runs `g1_bridge/` on G1.
- **Local HOMIE/WBC** owns legs and waist, merges PATH's `rt/arm_sdk` arm data,
  and remains the only `rt/lowcmd` publisher. Its current local checkout is a
  source snapshot without Git metadata; see [`../docs/homie_integration.md`](../docs/homie_integration.md).
- **Unitree `dex1_1_service`** owns the Dex1-1 serial connection. It publishes
  hand state and consumes the DDS command topics from `g1_bridge/`; see
  [`../docs/g1_installation.md`](../docs/g1_installation.md).
- **XRoboToolkit / PICO SDK** supplies the live PICO manager's controller input.
- **Remote SAM3 worker** runs on the selected inference host and sends camera
  segmentation and geometry to the Thor gateway.

The Homie snapshot and external runtime projects are not currently vendored in
this directory. Keep host-specific checkout paths and credentials in
deployment environment files, not PATH-G1 source. Record the exact source
revision or archive checksum when Homie is packaged for a deployment.
