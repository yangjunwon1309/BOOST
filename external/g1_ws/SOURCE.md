# G1 low-level workspace snapshot

Copied from `C:/studioprojects/url_g1_deploy/g1_ws` at source commit
`3d37aa43f88964fc4255857c840dd33dd08e6eda`.

The vendored `g1_ws` defines the target `G1_POLICY_ACTION` GCA1 protocol, the
461-byte `G1_ROBOT_STATE` GST1 packet, and the CRAM camera-frame header. GCA1
is a 29-DoF policy-action transport and is not interchangeable with BOOST's
PDU1 `G1_PATH_UPPER_COMMAND` arm packet.

The upstream `g1_lowlevel_bridge` publishes `rt/lowcmd`, including in its
default null-action smoke-test mode. Do not launch that executable with
DreamWaQ. BOOST's `boost/io/g1_bridge` remains the arm-only adapter: it accepts
PDU1, publishes `rt/arm_sdk`, and relays GST1 state. Unitree SDK2 is installed
separately and is not copied into this source snapshot.
