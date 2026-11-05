# Teleimager snapshot

Copied from `C:/studioprojects/url_g1_deploy/thirdparty/teleimager` at source
commit `3d37aa43f88964fc4255857c840dd33dd08e6eda`.

The server and the TIM2 recording protocol are kept at the target revision.
BOOST's G1 preset is `config/hbvcam_4m2214hd_g1.yaml`: head camera TIM2 on
55558 and D455 wrist camera TIM2 streams on 55559/55560. Install the server in
its own Python 3.10 environment outside the repository; the Thor-side relay
imports only this package's small `recording_protocol.py` module.
