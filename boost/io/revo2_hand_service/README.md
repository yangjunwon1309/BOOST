# Revo2 hand service adapter

The physical hand driver is Unitree's external
[`brainco_hand_service`](https://github.com/unitreerobotics/brainco_hand_service).
It connects Revo2 USB serial I/O to
`rt/brainco/{left,right}/cmd` and `rt/brainco/{left,right}/state`. Each hand
has six normalized values in `[0, 1]`, ordered Thumb, Thumb_aux, Index, Middle,
Ring, Pinky.

The shared command/state mapping and DDS transport live in
[`boost/control/hand_mppi`](../../control/hand_mppi/). On real hardware, Thor
publishes fresh retargeted targets and receives measured state. The G1 arm
bridge does not command Revo2. DreamWaQ remains the only hardware
`rt/lowcmd` writer; the hand service owns the separate hand serial links and
their command-loss watchdog.
