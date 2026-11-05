# DreamWaQ deployment snapshot

Copied from `C:/studioprojects/url_g1_deploy/thirdparty/dreamwaq_g1_29dof`
at source commit `3d37aa43f88964fc4255857c840dd33dd08e6eda`.

The deployment source (`deploy/`, `dreamwaq_interface/`, `README.md`,
`pyproject.toml`, and `LICENSE`) is retained. Only the runtime export
`checkpoints/new3_4_backpack_440000/exported/` is included; training checkpoints,
other policy exports, and upstream test directories are omitted. The selected
manifest is kept byte-for-byte with its `backpack_payload.json` metadata.

This export is the backpack-trained NEW3-4 policy. Its policy action is the
12 lower-body joints; waist and arms remain separate command owners. Keep the
export together with this source revision and do not substitute an export with
a different manifest/action contract.

The upstream hardware entry point is `deploy/run_dreamwaq.py`. BOOST's MuJoCo
path invokes the same `DreamWaQPolicy` and action compiler in-process, while
the hardware launcher invokes the upstream DDS runtime. No upstream code in
this snapshot is started implicitly by the sim launcher.

Selected export provenance (`new3_4_backpack_440000/exported`):

| File | SHA-256 |
| --- | --- |
| `backpack_payload.json` | `0cf679ffed05ec3f47585200073c2e9f55485c6a88f1f6219441ecd474a2874e` |
| `cenet.pt` | `1092cda452a5915fe950489b0236b83e2d6b8e2889797a26c17785f503c67c6f` |
| `cenet_norm.pt` | `d2c714dc195819eb3a8327009580c4650c3d01de0d5688bbec41e3527bf3e75d` |
| `manifest.json` | `edab7a217d74b0e64b23f9fcd896a63197113a8e4589e2599bfbc798cfd47ab7` |
| `policy.pt` | `e185edc6d695dc18f1cde0cfff43c2eb929f5b5ea8c7b842f6354fa5e2232afb` |

`configs/sim/dreamwaq_g1_mujoco.yaml` retains the target deployment joint
order, gains, limits, default pose, and arm-source behavior. Its sole runtime
setting change is `cmd_source: dds`, which allows the isolated simulator's
keyboard velocity topic to drive the policy.
