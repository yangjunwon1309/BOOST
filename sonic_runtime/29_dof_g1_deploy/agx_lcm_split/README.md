# AGX LCM Split Kit

This folder contains the AGX-side helper code for the split deployment:

```text
G1 onboard -> AGX
  G1_ROBOT_STATE

AGX -> G1 onboard
  G1_POLICY_ACTION
```

The packet format is fixed binary and shared with `29_dof_g1_deploy/g1_ws`.
Policy actions must be raw SONIC `action[29]` values. Do not send q targets,
gains, or scaled actions from the AGX. The G1 onboard bridge owns scaling,
joint ordering, timeout, and damping fallback.

## Copy To AGX

Copy this folder to the AGX, for example:

```bash
scp -r agx_lcm_split unitree@<agx-ip>:~/agx_lcm_split
```

If `gear_sonic_deploy` is already on the AGX, a convenient location is:

```text
~/GR00T-WholeBodyControl/agx_lcm_split
```

## Install LCM

Install on both the AGX and G1 onboard computer:

```bash
sudo apt update
sudo apt install -y liblcm-dev
```

## Build Smoke Tools On AGX

```bash
cd ~/agx_lcm_split
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

If your copy is under `~/GR00T-WholeBodyControl/agx_lcm_split`, use that path.

## First Network Test

Start the G1 onboard bridge in dry-run mode:

```bash
cd ~/29_dof_g1_deploy/g1_ws
./build/g1_lowlevel_bridge eth0 \
  --action-source lcm \
  --publish-lcm-state \
  --dry-run-actions
```

On the AGX, listen for robot state:

```bash
cd ~/agx_lcm_split
./build/g1_lcm_state_listener
```

Expected AGX output:

```text
[g1-lcm-state-receiver] First robot state. seq=...
[state-listener] packets=... seq=... age_ms=... mode=... q0=...
```

Then send zero actions from AGX:

```bash
./build/g1_lcm_action_smoke_sender
```

Expected G1 onboard output:

```text
[lcm-action] First action received. seq=0
[bridge] Dry-run action received seq=...
```

Because the robot is in `--dry-run-actions`, these packets do not move the
motors. The bridge keeps damping command active.

## Bidirectional Smoke Test

After the one-way tests pass, run a simultaneous receive/send test on the AGX.
This process subscribes `G1_ROBOT_STATE` while publishing zero
`G1_POLICY_ACTION` packets at 50 Hz:

```bash
./build/g1_lcm_bidirectional_smoke
```

With an explicit URL and a 30-second duration:

```bash
./build/g1_lcm_bidirectional_smoke "udpm://239.255.76.67:7667?ttl=1" 50 30
```

Expected AGX output:

```text
[bidirectional-smoke] sent_actions=50 received_states=100 last_state_seq=... state_age_ms=...
```

Expected G1 onboard output:

```text
[lcm-action] First action received. seq=0
[bridge] Dry-run action received seq=...
```

Keep `--dry-run-actions` enabled on the robot for this test.

## LCM URL

Usually the default LCM URL works when the AGX and G1 onboard are on the same
LAN. If multicast routing is ambiguous, use the same explicit URL on both
machines:

```bash
export LCM_DEFAULT_URL="udpm://239.255.76.67:7667?ttl=1"
```

Or pass it explicitly:

```bash
./build/g1_lcm_state_listener "udpm://239.255.76.67:7667?ttl=1"
./build/g1_lcm_action_smoke_sender "udpm://239.255.76.67:7667?ttl=1" 50
./build/g1_lcm_bidirectional_smoke "udpm://239.255.76.67:7667?ttl=1" 50 30
```

And on the robot:

```bash
./build/g1_lowlevel_bridge eth0 \
  --action-source lcm \
  --publish-lcm-state \
  --dry-run-actions \
  --lcm-url "udpm://239.255.76.67:7667?ttl=1"
```

## Integrate With `gear_sonic_deploy`

Add this include directory to the `gear_sonic_deploy` target that builds
`g1_deploy_onnx_ref`:

```cmake
target_include_directories(g1_deploy_onnx_ref PRIVATE
  /home/unitree/agx_lcm_split/include
)
target_link_libraries(g1_deploy_onnx_ref PRIVATE lcm)
```

Then add an action sender member to `G1Deploy`:

```cpp
#include "g1_lcm_action_sender.hpp"

std::unique_ptr<g1_lcm::ActionSender> lcm_action_sender_;
```

Initialize it after parsing CLI options:

```cpp
lcm_action_sender_ = std::make_unique<g1_lcm::ActionSender>(
    lcm_url, g1_lcm::kDefaultActionChannel);
if (!lcm_action_sender_->Start()) {
  throw std::runtime_error("Failed to start LCM action sender");
}
```

In `CreatePolicyCommand()`, after policy inference:

```cpp
std::array<float, g1_lcm::kNumMotors> raw_action{};
for (int i = 0; i < g1_lcm::kNumMotors; ++i) {
  raw_action[i] = floatarr[i];
}
lcm_action_sender_->Publish(raw_action, false);
```

For split mode, AGX must not publish `rt/lowcmd` directly. Keep the policy
inference and PICO/manager input running, but disable the AGX
`LowCommandWriter` path or guard it behind a `--split-lcm-action-output` flag.

## Full Split Robot State Integration

The helper `g1_lcm::RobotStateReceiver` receives `G1_ROBOT_STATE` from the G1
onboard bridge. Use it when you are ready to remove Unitree DDS state input from
the AGX process.

The packet contains:

```text
base_quat[4], base_gyro[3], base_accel[3]
torso_quat[4], torso_gyro[3], torso_accel[3]
q[29], dq[29], tau_est[29]
mode_machine
sequence, lowstate_sequence, send_time_us
```

Map these values into the same buffers currently filled by `LowStateHandler()`
and `imuTorsoHandler()` in `gear_sonic_deploy`.

Recommended order:

1. Verify `G1_ROBOT_STATE` receive rate on AGX.
2. Publish dry-run zero actions from AGX and confirm the robot receives them.
3. Add action sender to `gear_sonic_deploy` while keeping direct DDS state input.
4. Disable AGX `rt/lowcmd` publish in split mode.
5. Only after dry-run logs are clean, remove `--dry-run-actions` on the robot.
