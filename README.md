# PATH-G1

PATH-G1 is the research workspace for intention-driven teleoperation and
planning-assisted manipulation on Unitree G1.

The repository keeps the modified SONIC runtime inside `sonic_runtime/`, while
GMR and cuRoboV2 remain external installed dependencies.

- `sonic_runtime/gear_sonic_deploy/`: canonical SONIC policy/deploy runtime,
  including `ee_qp` and `g1_deploy_onnx_ref_PATH.cpp`.
- `sonic_runtime/gear_sonic/`: canonical modified SONIC Python package.
- `path_core/`: PATH state machine, intention interfaces, reference-provider
  APIs, messaging, and runtime logging.
- `path_planning/`: cuRoboV2 planning server and B-spline goal utilities.
- `path_streaming/`: host-side SMPL/GMR replay, Pico manager, and recorder
  servers.
- `tools/`: runtime/debug utilities.
- `analysis/`: FK error and report-generation scripts.
- `configs/`: robot, QP, runtime, and state-machine configuration.

Heavy external dependencies are not vendored here. Configure their installed
paths in `configs/runtime/local.env` or by exporting environment variables.

## Host Python Environment

```bash
cd /home/url/PATH-G1
source configs/runtime/local.env
source "$PATH_GROOT_ROOT/.venv_sim/bin/activate"
```

## Start PATH/cuRobo B-spline Server

```bash
cd /home/url/PATH-G1
./scripts/run_curobo_server.sh
```

Equivalent expanded command:

```bash
python -u path_planning/path_curobo_planning_server.py \
  --curobo-root "$PATH_CUROBO_ROOT" \
  --solver trajopt \
  --robot unitree_g1_29dof_retarget.yml \
  --num-seeds 4 \
  --lbfgs-iters 25 \
  --use-cuda-graph \
  --output-port 5560 \
  --goal-port 5561
```

## Start Pico / SMPL Stream Server

```bash
cd /home/url/PATH-G1
./scripts/run_pico_manager.sh
```

For recorded SMPL replay:

```bash
cd /home/url/PATH-G1
./scripts/run_recorded_smpl_server.sh /path/to/recording_dir --start-planner
```

## Start PATH SONIC Deploy in MuJoCo

Run the policy inside the deploy Docker. On Thor, start Docker from the host:

```bash
docker run --rm -it \
  --runtime=nvidia --gpus all \
  --privileged --net=host \
  -e NVIDIA_DISABLE_REQUIRE=true \
  -v ~/:/workspace \
  -v ~/trt_clean:/opt/trt_clean \
  -v /usr/lib/aarch64-linux-gnu/nvidia:/host_nvidia_libs \
  -w /workspace/ \
  g1-deploy-lcm:latest /bin/bash
```

Inside Docker:

```bash
cd /workspace/PATH-G1
./path_deploy/scripts/run_path_mujoco.sh
```

Equivalent runtime directory:

```bash
cd /workspace/PATH-G1/sonic_runtime/gear_sonic_deploy

PATH_BSPLINE_ENABLED=1 \
PATH_BSPLINE_PORT=5560 \
PATH_BSPLINE_TOPIC=curobo_bspline \
bash teleop_sonic_mujoco_ik.sh
```

## Send a Small Test Goal

```bash
cd /home/url/PATH-G1
./scripts/send_left_wrist_start_offset_goal.sh
```

