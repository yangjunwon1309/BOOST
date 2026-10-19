# G1 deployment installation

## Selected HOMIE source

Use the G1-local HOMIE checkout that contains `deploy/run_homie.py`,
`deploy/robot_control/configs/g1_homie.yaml`, and both `ckpts/stand.onnx` and
`ckpts/walk.onnx`. The PATH-G1 `homie-g1-snapshot` branch also carries the
deployment snapshot under `external/homie_g1_29dof/`; its two checkpoint hashes
match the current Windows source tree. If HOMIE is already cloned on G1 with
these files, use that checkout and skip transfer and installation steps below.

The run commands later in this guide assume
`/home/unitree/00_workspace/homie_g1_29dof/deploy`. If your clone is elsewhere,
run them from that clone's `deploy/` directory. If you use the bundled snapshot,
run them from `$HOME/PATH-G1/external/homie_g1_29dof/deploy`; its checkpoints
are already under the matching `../ckpts/` path. Use the
`homie-g1-snapshot` branch for the PATH-G1 runtime checkout on G1, Thor, and the
remote SAM3 worker so all three use the same deployment code. Keep the separate
HOMIE clone as the `rt/lowcmd` controller source.

The local configuration selects `arm_source: "arm_sdk"`. HOMIE combines PATH's
14 arm targets with its 15 leg/waist targets and must remain the only
`rt/lowcmd` publisher. Do not run upstream
`HomieDeploy/unitree_sdk2/g1_control` alongside it; that binary consumes a
different arm-input protocol. See [`homie_integration.md`](homie_integration.md).

## Transfer local HOMIE from Windows

Replace `<G1_IP>` with the G1 computer's address. The commands below use the
usual `unitree` account and `/home/unitree/00_workspace` install directory.

```powershell
$g1 = "unitree@<G1_IP>"
$bundle = Join-Path $env:TEMP "homie_g1_29dof.tar.gz"
tar -czf $bundle -C "C:\studioprojects\homie_g1_29dof" "homie_g1_29dof"
Set-Location $env:TEMP
scp .\homie_g1_29dof.tar.gz "${g1}:/home/unitree/00_workspace/"
ssh $g1 "mkdir -p /home/unitree/00_workspace && tar -xzf /home/unitree/00_workspace/homie_g1_29dof.tar.gz -C /home/unitree/00_workspace"
```

The bundle includes the `stand.onnx` and `walk.onnx` checkpoints from the local
folder's `ckpts/` directory.

## Install the local HOMIE Python environment on G1

The commands match the local HOMIE deployment guide. They assume Conda is
installed on G1 and the robot's Unitree SDK2 Python source is at
`/home/unitree/00_workspace/unitree_sdk2_python`.

If that SDK2 Python source is not already present, install the upstream source
at the expected path first:

```bash
cd /home/unitree/00_workspace
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
git -C unitree_sdk2_python checkout 814556d15970dd2ecf1c9984e845ca02ab07e206
```

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n homie python=3.8 -y
conda activate homie
python -m pip install cyclonedds==0.10.2 "numpy<2" onnxruntime==1.18.0 pyyaml opencv-python
python -m pip install -e /home/unitree/00_workspace/unitree_sdk2_python --no-deps
```

If `pip install cyclonedds==0.10.2` cannot find the native CycloneDDS library,
build the 0.10.x C library first, set `CYCLONEDDS_HOME`, and retry the pip
install. Unitree's SDK2 Python installation notes describe this native-library
step.

```bash
sudo apt update
sudo apt install -y cmake g++ build-essential git
cd /home/unitree/00_workspace
git clone https://github.com/eclipse-cyclonedds/cyclonedds.git
git -C cyclonedds checkout 5041f3560c088c99e5088b2b8520b69169621196
cmake -S cyclonedds -B cyclonedds/build -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build cyclonedds/build --parallel 4
sudo cmake --install cyclonedds/build
sudo ldconfig
export CYCLONEDDS_HOME=/usr/local
conda activate homie
python -m pip install cyclonedds==0.10.2 "numpy<2" onnxruntime==1.18.0 pyyaml opencv-python
```

## Install PATH-G1 and the G1 bridge

On G1, install the compiler and LCM development package, then clone the deploy
branch. The branch URL is `https://github.com/yangjunwon1309/PATH-G1`.

```bash
sudo apt update
sudo apt install -y git cmake g++ build-essential pkg-config liblcm-dev \
  libyaml-cpp-dev libeigen3-dev libboost-all-dev libfmt-dev
git clone --branch homie-g1-snapshot --single-branch \
  https://github.com/yangjunwon1309/PATH-G1.git "$HOME/PATH-G1"
```

The bridge builds against Unitree's C++ SDK2 source. The commands below pin
the SDK revision checked for this deployment document (`d1abf00`).

```bash
mkdir -p "$HOME/00_workspace"
git clone https://github.com/unitreerobotics/unitree_sdk2.git \
  "$HOME/00_workspace/unitree_sdk2"
git -C "$HOME/00_workspace/unitree_sdk2" \
  checkout d1abf001220d3e5f59e00a0c4aaa595e00089a4a
cd "$HOME/PATH-G1"
UNITREE_SDK2_ROOT="$HOME/00_workspace/unitree_sdk2" ./g1_bridge/build.sh
```

Find the G1 network interface with `ip -brief addr`. Dry-run the bridge first;
this receives state and LCM packets but publishes no arm or hand targets.
Dex1 DDS initialization is skipped in dry-run mode.

```bash
cd "$HOME/PATH-G1"
G1_NETWORK_INTERFACE=eth0 ./g1_bridge/run.sh
```

## Run the controllers

Start and leave `dex1_1_gripper_server` running on the computer connected to
the hands using the commands in the next section before launching the bridge
live. Use separate terminals on G1. Start the local HOMIE WBC as the only
`rt/lowcmd` publisher. Start the PATH bridge in live mode for `rt/arm_sdk` and
Dex1 hand output. Live Dex1 startup reads both hand state topics and opens the
grippers; PATH grip flags then set close/open targets. Stop each dry-run with
Ctrl+C before starting that process in live mode.

The checked-in HOMIE config selects `cmd_source: "udp"`. The HOMIE process on
G1 receives Thor's locomotion `cmd_vel` at `192.168.123.164:15001` and accepts
packets only from Thor at `192.168.123.2`; the payload carries `vx`, `vy`,
`wz`, `session`, and `seq`. This is the intended high-level command path into
HOMIE, not a packet to the Unitree handheld remote. The handheld sticks are
read separately by HOMIE and override UDP input. The PATH-G1 checkout contains
the receiver; the existing Thor-side sender must target this address and port.
Verify the two IPs against the actual network configuration.

```bash
cd /home/unitree/00_workspace/homie_g1_29dof/deploy
bash run_homie.sh --net eth0 --dry-run
# Press Ctrl+C after checking the dry-run output.
bash run_homie.sh --net eth0
```

```bash
cd "$HOME/PATH-G1"
G1_NETWORK_INTERFACE=eth0 ./g1_bridge/run.sh
# Press Ctrl+C after checking LCM/state traffic, then start live mode.
G1_NETWORK_INTERFACE=eth0 G1_BRIDGE_LIVE=1 ./g1_bridge/run.sh
```

The bridge defaults to the Dex1 hand backend. To run on a G1 without Dex1
hands, pass `--hand-backend none` to `run.sh`. Do not start another process
that publishes `rt/lowcmd` while local `run_homie.py` is running.

## Install the Dex1-1 serial-to-DDS service

Install and run Unitree's [`dex1_1_service`](https://github.com/unitreerobotics/dex1_1_service)
on the computer physically connected to the Dex1 serial cables (Unitree's
README describes PC2 / Jetson Orin NX). This service owns serial I/O; the
PATH bridge communicates with it over Unitree DDS using the `rt/dex1/*`
topics documented in [`protocols.md`](protocols.md). The service and bridge
must use DDS domain 0 and network interfaces that can exchange DDS traffic.

```bash
sudo apt update
sudo apt install -y build-essential cmake g++ libyaml-cpp-dev libeigen3-dev \
  libboost-all-dev libfmt-dev libserialport-dev libspdlog-dev
mkdir -p "$HOME/00_workspace"
git clone https://github.com/unitreerobotics/unitree_sdk2 \
  "$HOME/00_workspace/unitree_sdk2"
git -C "$HOME/00_workspace/unitree_sdk2" \
  checkout d1abf001220d3e5f59e00a0c4aaa595e00089a4a
cd "$HOME/00_workspace/unitree_sdk2"
mkdir -p build
cd build
cmake ..
sudo make install
cd "$HOME/00_workspace"
git clone https://github.com/unitreerobotics/dex1_1_service \
  "$HOME/00_workspace/dex1_1_service"
git -C "$HOME/00_workspace/dex1_1_service" \
  checkout ced3216290bb22dae356213993786beeaa7143ab
cd "$HOME/00_workspace/dex1_1_service"
mkdir -p build
cd build
cmake ..
make -j6
cd ../bin
sudo ./dex1_1_gripper_server --network eth0
```

Replace `eth0` with the interface on that computer that shares DDS traffic with
G1. The G1 bridge also takes its DDS interface from `G1_NETWORK_INTERFACE`.
Do not run the service's sine-wave test publisher during operation; it sends
competing commands to the same hand topics. For service verification, Unitree
provides `sudo ./test_dex1_1_gripper_server --network eth0 -l -r` from the same
`bin/` directory. Stop that test process before starting the PATH bridge live.

## Where the cameras and `rt/lowcmd` go

The wrist camera is captured by PATH on Thor. SAM3 inference may run in the
remote worker; surface fitting and belief assistance stay in the PATH
perception flow. The G1 bridge does not capture or stream images. Its body-arm
output is `rt/arm_sdk`, and its hand outputs are `rt/dex1/*/cmd`. Local HOMIE
alone sends the merged whole-body command to `rt/lowcmd`.
