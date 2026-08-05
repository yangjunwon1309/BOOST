# BrainCo REVO2 MuJoCo backend

This is an opt-in variant. The original DEX3 MJCF, seven-command DDS path,
logging, and tests remain unchanged.

## Model layout

- G1 body policy/action: unchanged 29 DoF
- REVO2 command count: 6 per hand
- REVO2 physical joint count: 11 per hand
- Actuated: thumb metacarpal, thumb proximal, index, middle, ring, pinky proximal
- Coupled: thumb/index/middle/ring/pinky distal via MJCF equality constraints
- Right-hand passive tip hinges from the supplied standalone model are fixed so
  both hands expose the same 11-joint state layout. Tip collision meshes remain.

Generated default-scene files:

```text
gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_revo2.xml
gear_sonic/data/robot_model/model_data/g1/scene_43dof_revo2.xml
```

Equivalent deploy/torres variants are under `gear_sonic_deploy/g1/`. Regenerate
all variants after changing mount transforms or the source hand XML:

```bash
cd /home/url/PATH-G1
python3 sonic_runtime/tools/build_revo2_mjcf.py \
  --revo2-root /home/url/Downloads/Revo2_xml
```

## DDS command convention

The active REVO2 topics use `unitree_go::msg::dds_::MotorCmds_` and
`MotorStates_`:

```text
rt/brainco/left/cmd       rt/brainco/left/state
rt/brainco/right/cmd      rt/brainco/right/state
```

Each message has six normalized `[0, 1]` entries in official BrainCo order:

```text
[Thumb, Thumb_aux, Index, Middle, Ring, Pinky]
```

The static binary adapter maps:

```text
open  = [0,    0, 0, 0, 0, 0]
close = [0.70, 1, 1, 1, 1, 1]
```

The thumb opposition limit is intentionally conservative for initial contact
testing. Both poses can be changed in
`g1_29dof_sonic_model12_revo2.yaml`.

For compatibility, the REVO2 simulator also listens to legacy
`rt/dex3/{left,right}/cmd`. It reduces the DEX3 seven-vector to the same binary
flag used by the real LCM bridge (`abs(q[3]) > 0.10`) and then emits the static
REVO2 pose internally. REVO2 state is published only on `rt/brainco/*/state`.

## Selecting the backend

Pass the variant through the existing simulation launcher:

```bash
cd /home/url/PATH-G1
./start_sonic_sim.sh --wbc-version sonic_model12_revo2
```

The legacy DEX3 model remains the default when `--wbc-version` is omitted.
This command is documentation only; model generation and unit validation do
not start the SONIC policy or MuJoCo process.

## Validation

```bash
cd /home/url/PATH-G1
source configs/runtime/local.env
source "$PATH_GROOT_ROOT/.venv_sim/bin/activate"
python sonic_runtime/gear_sonic/tests/test_revo2_hand_backend.py -v
```
