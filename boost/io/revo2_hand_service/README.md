# Revo2 hand service adapter

The hand driver is the external Unitree
[`brainco_hand_service`](https://github.com/unitreerobotics/brainco_hand_service)
project; its source is not copied into BOOST. It connects the Revo2 USB-serial
driver to `rt/brainco/{left,right}/cmd` and
`rt/brainco/{left,right}/state`. Each hand has six normalized values in
`[0, 1]`, ordered Thumb, Thumb_aux, Index, Middle, Ring, Pinky.

The BrainCo command/state mapping and DDS transport used by the active hand
retargeter live in `boost/control/hand_mppi/`. On real hardware, Thor publishes
fresh targets and receives measured state on these topics. The G1 arm bridge
under `../g1_bridge/` does not command Revo2. HOMIE remains the only
`rt/lowcmd` writer; hand commands use the separate BrainCo topics.
