# Revo2 hand service adapter

The physical hand driver is Unitree's external [`brainco_hand_service`](https://github.com/unitreerobotics/brainco_hand_service). It connects Revo2 USB serial I/O to `rt/brainco/{left,right}/cmd` and `rt/brainco/{left,right}/state`. Each hand has six normalized values in `[0, 1]`, ordered Thumb, Thumb_aux, Index, Middle, Ring, Pinky.

The command/state mapping and DDS transport live in `boost/control/hand_mppi/`. The G1 arm adapter handles the older PATH/Dex1 contract and does not drive physical Revo2. Revo2 simulation and retargeting are integrated; physical Revo2 service deployment remains separate from the DreamWaQ body controller. DreamWaQ alone owns hardware `rt/lowcmd`; hand commands use their own DDS topics.
