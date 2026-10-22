# Revo2 hand service adapter

The hand driver is the external Unitree
[`brainco_hand_service`](https://github.com/unitreerobotics/brainco_hand_service)
project; its source is not copied into BOOST. It connects the Revo2 USB-serial
driver to `rt/brainco/{left,right}/cmd` and
`rt/brainco/{left,right}/state`. Each hand has six normalized values in
`[0, 1]`, ordered Thumb, Thumb_aux, Index, Middle, Ring, Pinky.

The BrainCo command/state mapping and DDS transport shared with the future
retargeter live in `boost/control/hand_mppi/`. The PATH G1 bridge under
`../g1_bridge/` still only supports the older Dex1 packet and does not implement
this interface. HOMIE remains the only `rt/lowcmd` writer; hand commands are
published on the separate BrainCo topics.
