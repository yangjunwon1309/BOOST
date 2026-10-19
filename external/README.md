# External runtime components

PATH-G1 contains its own arm control, perception integration, robot model, and
G1 arm bridge. These components remain separate deployments:

- **Unitree SDK2** builds and runs the G1-local `g1_bridge/` process.
- **HOMIE/WBC** owns legs and waist from its own repository and configuration.
- **Dex1 service** owns hand hardware and consumes hand intent separately.
- **XRoboToolkit / PICO SDK** supplies the live PICO manager's controller input.
- **Remote SAM3 worker** runs on the selected inference host; it sends camera
  segmentation and geometry data to the Thor gateway.

This directory does not vendor these projects. Keep host-specific checkout
paths and credentials in deployment environment files, not in PATH-G1 source.
HOMIE fork, commit, and G1 topic configuration are recorded with its deployment.
