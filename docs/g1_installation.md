# G1 installation

The active G1 body controller is the backpack-trained DreamWaQ runtime from `external/dreamwaq_g1_29dof/`. Its deployment package keeps the target config filename `g1_homie.yaml`; this is the url_g1_deploy config, not BOOST's retired HOMIE runtime. The selected export is `new3_4_backpack_440000/exported`.

From the BOOST checkout on the G1 host, install the two isolated Python environments:

```bash
bash scripts/setup_dreamwaq_runtime.sh
bash scripts/setup_teleimager_runtime.sh
```

Build `boost/io/g1_bridge/` against the installed Unitree SDK2 C++ tree as described in that directory's README. Then start Teleimager as the single camera owner:

```bash
bash scripts/run_g1_teleimager.sh
```

Start the G1 arm/state adapter and DreamWaQ together from the same checkout:

```bash
BOOST_ALLOW_HARDWARE_CONTROL=1 \
G1_NETWORK_INTERFACE=eth0 \
bash scripts/run_g1_dreamwaq_controller.sh
```

The DreamWaQ runtime begins in standing mode and follows the target deployment START then A enable procedure. Press X to toggle standing/walking. The script starts the PATH adapter on `rt/arm_sdk` and DreamWaQ as the sole `rt/lowcmd` owner. Do not start `g1_ws/g1_lowlevel_bridge`, a HOMIE runtime, or another lowcmd writer alongside it.

See [`dreamwaq_integration.md`](dreamwaq_integration.md) for runtime pins, CPU partition defaults, network interfaces, Thor-side TIM2 relay, and the separate MuJoCo launch path. The physical Dex1 and Revo2 services are not started by these commands.
