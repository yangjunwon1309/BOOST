# External runtime components

These source snapshots are pinned to url_g1_deploy commit `3d37aa43f88964fc4255857c840dd33dd08e6eda` and keep source metadata in each directory's `SOURCE.md`.

- [`dreamwaq_g1_29dof/`](dreamwaq_g1_29dof/) contains the target DreamWaQ deployment runtime, the selected `new2_3_backpack_lower8cm_299999` export, and the retained `new3_4_backpack_440000` export. BOOST defaults hardware and MuJoCo to NEW2-3. The full deployment runtime owns hardware `rt/lowcmd`; the sim calls its policy and action compiler in-process.
- [`g1_ws/`](g1_ws/) records the target DDS and packet schemas. Its low-level bridge is not launched with DreamWaQ because it publishes the same `rt/lowcmd` topic.
- [`teleimager/`](teleimager/) contains the TIM2 image server/protocol and selected HBVCAM/D455 G1 camera preset. The server is installed separately from DreamWaQ.
- [`brainco-description/`](brainco-description/) pins the Revo2 hand assets used by the MuJoCo model.
- Unitree SDK2 C++ and Python components are installed as runtime dependencies; they are not copied from a complete upstream workspace.
- Televuer and existing PATH Python dependencies remain pinned in Pixi for the desktop simulation/teleop environment.

DreamWaQ and Teleimager virtual environments default to `$HOME/.local/share/boost/external/venvs/`. No camera service, low-level G1 controller, or external runtime starts automatically when the repository is imported or when MuJoCo is launched.
