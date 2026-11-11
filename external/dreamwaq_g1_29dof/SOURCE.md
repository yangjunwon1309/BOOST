# DreamWaQ deployment snapshot

Copied from `C:/studioprojects/url_g1_deploy/thirdparty/dreamwaq_g1_29dof`
at source commit `3d37aa43f88964fc4255857c840dd33dd08e6eda`.

The deployment source (`deploy/`, `dreamwaq_interface/`, `README.md`,
`pyproject.toml`, and `LICENSE`) is retained. The runtime exports
`checkpoints/new2_3_backpack_lower8cm_299999/exported/` and
`checkpoints/new3_4_backpack_440000/exported/` are included byte-for-byte with
their `backpack_payload.json` metadata. The original training checkpoints,
other policy exports, and upstream test directories are omitted. NEW2-3 is
BOOST's default hardware and MuJoCo export; NEW3-4 remains available as an
alternate.

Both exports are backpack-trained policies with 12 lower-body actions; waist
and arms remain separate command owners. NEW2-3 uses the 99-dimensional
reference observation layout without an explicit standing/walking mode input.
NEW3-4 includes that explicit mode input. Keep each export together with this
source revision and do not substitute an export with a different
manifest/action contract.

The upstream hardware entry point is `deploy/run_dreamwaq.py`. BOOST's MuJoCo
path invokes the same `DreamWaQPolicy` and action compiler in-process, while
the hardware launcher invokes the upstream DDS runtime. No upstream code in
this snapshot is started implicitly by the sim launcher.

Default export provenance (`new2_3_backpack_lower8cm_299999/exported`):

| File | SHA-256 |
| --- | --- |
| `backpack_payload.json` | `d2ecf83f4960f7f5ce76db497fa7650603bfb9f061b497bc3458a31b57b853ef` |
| `cenet.pt` | `b8c68b9c3cd30f56fd45c76fa48a24ee0f7830cb71ab259ed6050943a578431d` |
| `cenet_norm.pt` | `8eb2574f056108e50b34195712a855b4d6c4667b0f8b197373dca1930e0451eb` |
| `manifest.json` | `a8ed387cc909cc5fbbd2f69853ec95459f11d971ae0410c16324a261c3fee0b2` |
| `policy.pt` | `f35c1e549a155089b3a40eeb282f7ef2c4617cfd7b27f041ae1ba78da68f8ea6` |

Alternate export provenance (`new3_4_backpack_440000/exported`):

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
