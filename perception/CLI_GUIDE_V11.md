# PATH-G1 v11 bottle-open launch guide

This guide is for the current `PATH-G1` checkout.  Paste one **terminal block**
into one terminal on the named machine; do not combine blocks in a single
terminal.  Start the long-running processes in the order shown and leave each
one running.

The common transports are fixed for this setup:

```bash
G1_LCM_URL='udpm://239.255.76.67:7667?ttl=1'
WIFI_LCM_URL='udpm://239.255.77.67:7667?ttl=1'
```

## Important current-version rules

- Use the **wrist D405** launcher below.  Do not use the old head-D435i
  `run_thor_remote_gateway.sh` for this experiment.
- Do not export fixed `D405_VIDEO_DEVICES` or `D405_USB_DEVICES`.  USB and
  `/dev/videoN` numbers change after reconnect/reboot.  `--auto-devices`
  discovers the two D405s immediately before Docker starts.
- The currently deployed `run_homie.py` accepts `--net` and
  `--writer-rate-hz`; do **not** pass
  `--arm-max-velocity-rad-s` or `--arm-tau-mode`.
- Only one process may own the D405 V4L2 nodes.  If the gateway reports an
  owner conflict, stop that camera-owning service before retrying.

## 1. G1: Dex1 gripper server (Terminal G1-1)

```bash
cd ~/dex1_1_service
sudo ./bin/dex1_1_gripper_server --network eth0
```

## 2. G1: low-level bridge (Terminal G1-2)

```bash
cd ~/URL/29_dof_g1_deploy/g1_ws
./build/g1_lowlevel_bridge eth0 \
  --action-source lcm \
  --dds-output-mode arm-sdk \
  --hand-backend dex1 \
  --publish-lcm-state \
  --lcm-url 'udpm://239.255.76.67:7667?ttl=1' \
  --lcm-action-channel G1_PATH_MOTOR_COMMAND \
  --lcm-state-channel G1_ROBOT_STATE
```

## 3. G1: direct-LCM mux (Terminal G1-3)

```bash
cd ~/URL/29_dof_g1_deploy/g1_ws
bash tools/run_path_direct_lcm_bridge.sh
```

## 4. G1: Homie policy writer (Terminal G1-4)

```bash
source /home/unitree/miniconda3/etc/profile.d/conda.sh
conda activate homie
cd /home/unitree/00_workspace/g1_deploy/thirdparty/homie_g1_29dof/deploy
python run_homie.py \
  --net eth0 \
  --writer-rate-hz 250
```

The old `--arm-max-velocity-rad-s 0 --arm-tau-mode forward` suffix is not
supported by this deployed CLI and must not be added.

## 5. Thor: SAM3 remote worker (Terminal T-1)

```bash
cd ~/PATH-G1/perception
export WIFI_LCM_URL='udpm://239.255.77.67:7667?ttl=1'
export G1_PERCEPTION_CONTAINER_NAME='g1-perception-wrist-sam3-v11'

./tools/run_thor.sh python -m remote_offload.remote_sam3_wrist_d405_worker_v11 \
  --wifi-lcm-url "$WIFI_LCM_URL" \
  --backend tensorrt \
  --trt-engine /models/sam3_trt/sam3_fp16_4out.engine \
  --concepts object \
  --extrinsics config/wrist_d405_extrinsics_v11_translation_calibrated.yaml \
  --belief-config config/pipeline.yaml \
  --pairing-max-residual-ms 15 \
  --pairing-max-error-m 0.008
```

## 6. Thor: two-wrist D405 capture and belief gateway (Terminal T-2)

```bash
cd ~/PATH-G1/perception
export G1_LCM_URL='udpm://239.255.76.67:7667?ttl=1'
export WIFI_LCM_URL='udpm://239.255.77.67:7667?ttl=1'
export G1_PERCEPTION_CONTAINER_NAME='g1-perception-wrist-d405-v11'
unset D405_VIDEO_DEVICES D405_USB_DEVICES

./tools/run_thor_wrist_d405_gateway_v11.sh \
  --auto-devices \
  --g1-lcm-url "$G1_LCM_URL" \
  --wifi-lcm-url "$WIFI_LCM_URL" \
  --config config/wrist_d405_v11.yaml \
  --belief-config config/pipeline.yaml \
  --max-hz 8 \
  --camera-recovery-initial-delay-ms 300 \
  --camera-recovery-max-delay-ms 2000 \
  --belief-inference \
  --belief-input-source operator_command \
  --belief-command-max-delta-ms 25 \
  --belief-command-divergence-max-m 0.20 \
  --ee-target-publishing \
  --log-level INFO
```

It must print both `Auto-detected D405_VIDEO_DEVICES=...` and
`Auto-detected D405_USB_DEVICES=...`.  If either is missing, do not replace it
with an old hard-coded device number; diagnose the physical USB/V4L2 mapping.

## 7. G1 or control host: PATH state-machine bridge (Terminal C-1)

```bash
cd ~/PATH-G1
export PATH_SHARED_POSITION_MODE='replace'
unset PATH_SHARED_POSITION_ALPHA PATH_SHARED_POSITION_ALPHA_SOURCE
bash path_deploy/scripts/run_path_direct_lcm.sh
```

This launches the current state-machine bridge only; the separate G1 local
bridge from Terminal G1-2 and mux from Terminal G1-3 must already be running.

## 8. XR input: choose one manager

### Pico manager (Terminal XR-1)

`configs/runtime/local.env` supplies the virtual environment and XRoboToolkit
paths, so do not copy stale manual `PYTHONPATH`/`LD_LIBRARY_PATH` exports.

```bash
cd ~/PATH-G1
bash scripts/run_pico_manager.sh
```

### Galaxy XR manager (Terminal XR-2; use instead of Pico when applicable)

```bash
cd ~/PATH-G1
export PATH_TELEVUER_ROOT='/home/url/00_workspace/xr_teleop/teleop/televuer'
export PATH_TELEVUER_CERT="$HOME/.config/xr_teleoperate/cert.pem"
export PATH_TELEVUER_KEY="$HOME/.config/xr_teleoperate/key.pem"
bash scripts/run_galaxy_xr_manager.sh
```

## 9. Optional: Thor web view (Terminal T-3)

```bash
cd ~/PATH-G1/perception
export WIFI_LCM_URL='udpm://239.255.77.67:7667?ttl=1'
export SAM3_DOCKER_IMAGE='g1-perception-sam3:thor-jp71'
export SAM3_WEB_HOST='0.0.0.0'
export SAM3_WEB_PORT='8082'
export SAM3_IPOPT_VISUAL_CAMERA_Z_OFFSET_M='0.25'
./tools/run_remote_web_wrist_d405_v11.sh
```

## 10. Optional: record one trial (Terminal R-1)

Start this after the gateway is publishing.  It records until `Ctrl-C`, then
writes the matching `.summary.json` next to the CSV.

```bash
cd ~/PATH-G1
python3 perception/tools/record_wrist_experiment_v11.py \
  --wifi-lcm-url 'udpm://239.255.77.67:7667?ttl=1' \
  --output record/bottle_open/nonexpert/no_orientation/trial_001.csv  \
  --x-toggle
```

For repeated controller-X start/stop recording, add `--x-toggle`.  Do not
reuse an existing output filename without `--overwrite`.

## Ablation substitutions

For each ablation, change the **same belief configuration** in both Thor
Terminal T-1 (`--belief-config`) and Terminal T-2 (`--belief-config`).  Restart
both Thor processes after the change.

| ablation | Thor belief config | C-1 environment |
| --- | --- | --- |
| PATH baseline | `config/pipeline.yaml` | `PATH_SHARED_POSITION_MODE=replace` |
| No jump diffusion | `config/ablations/pipeline_v11_no_jump_diffusion.yaml` | `PATH_SHARED_POSITION_MODE=replace` |
| No orientation | `config/ablations/pipeline_v11_no_orientation_belief.yaml` | `PATH_SHARED_POSITION_MODE=replace` |
| Position Only | `config/pipeline.yaml` | `PATH_SHARED_POSITION_MODE=alpha`, `PATH_SHARED_POSITION_ALPHA_SOURCE=fixed`, `PATH_SHARED_POSITION_ALPHA=0.00` |
| Alpha blend | `config/pipeline.yaml` | `PATH_SHARED_POSITION_MODE=alpha`, `PATH_SHARED_POSITION_ALPHA_SOURCE=fixed`, `PATH_SHARED_POSITION_ALPHA=0.50` |

For the two alpha modes, replace the Terminal C-1 environment block before
starting the state machine:

```bash
# Position Only
export PATH_SHARED_POSITION_MODE='alpha'
export PATH_SHARED_POSITION_ALPHA_SOURCE='fixed'
export PATH_SHARED_POSITION_ALPHA='0.00'
```

```bash
# Alpha blend
export PATH_SHARED_POSITION_MODE='alpha'
export PATH_SHARED_POSITION_ALPHA_SOURCE='fixed'
export PATH_SHARED_POSITION_ALPHA='0.50'
```

```bash
# Alpha blend + LPF: filter only the final blended position target.
# 0.30 is a basic first-order EWMA setting (new-target weight).
export PATH_SHARED_POSITION_MODE='alpha'
export PATH_SHARED_POSITION_ALPHA_SOURCE='fixed'
export PATH_SHARED_POSITION_ALPHA='0.50'
export PATH_SHARED_POSITION_ALPHA_LPF='1'
export PATH_SHARED_POSITION_ALPHA_LPF_ALPHA='0.30'
```

`PATH_SHARED_POSITION_ALPHA_LPF` defaults to off, so the existing alpha
ablation is unchanged. When enabled, the first B/Y-gated alpha tick is seeded
at the VR position and subsequent blended targets use
`p_lpf = 0.70 * p_lpf(previous) + 0.30 * p_alpha`. The LPF resets as soon as
position authority is released; it does not filter normal VR or replace-mode
targets.

`No jump diffusion` and `No orientation` do not require a different G1 bridge
command; only their paired Thor belief-config path changes.
