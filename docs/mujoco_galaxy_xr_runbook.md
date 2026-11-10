# MuJoCo + Galaxy XR runbook

This runbook covers the local G1 + Revo2 MuJoCo teleoperation stack launched by
`pixi run -e sim mujoco-hand-xr`. It runs the simulator with DreamWaQ evaluated
in-process, the PATH arm bridge and IK, Televuer, and the Revo2 hand MPPI runner.
It does not start a hardware `rt/lowcmd` writer.

## One-time desktop setup

From the BOOST repository root:

```bash
git submodule update --init --recursive
export PIXI_FROZEN=true
pixi install -e sim
pixi run -e sim setup-sim
pixi install -e hand
```

On first launch, the XR launcher creates a Televuer TLS certificate and key
under `~/.config/xr_teleoperate/` if no readable pair is configured. The
headset may need to trust or import `cert.pem`.

## Start the live XR simulation

Run this in a terminal and leave it open while teleoperating:

```bash
export PIXI_FROZEN=true
pixi run -e sim mujoco-hand-xr
```

The MuJoCo viewer opens with the configured table-room scene. The desktop and
Galaxy XR headset must be on the same Wi-Fi/LAN. Find the desktop's LAN address
with:

```bash
hostname -I
```

On Galaxy XR, open the secure URL using the desktop's Wi-Fi address:

```text
https://<DESKTOP-LAN-IP>:8012
```

For example, if the desktop address is `192.168.1.42`, open
`https://192.168.1.42:8012`. Do not use `localhost` or `127.0.0.1` on the
headset; those refer to the headset itself. Port 8012 is Televuer's HTTPS web
service. Port 5566 carries Galaxy XR hand pose and landmark messages to PATH.

Wait until the headset reports live hand tracking, then press `r` in the PATH
terminal to start or resume arm following. PATH first confirms the G1 is in its
Q_READY pose. `[PATH-state]` should then report `teleop_tracking` and
`xr_wrist_fresh=True`. Press `s` to start a wrist experiment CSV and again to
stop and save it; press `q` to stop and exit. Revo2 finger targets are handled
by hand MPPI in parallel. Stale XR frames are rejected so an old target is not
replayed after tracking drops.

With the MuJoCo viewer focused, locomotion keys are W/S (forward/back), A/D
(left/right), Q/R (yaw), O/P (speed), and Space (stop). Direction keys latch
until reversed or stopped.

## Stop the stack

Press **Ctrl+C** in the launcher terminal. The launcher stops the simulator,
Televuer, PATH, and hand-MPPI child processes together.

## If startup says ports are already in use

Inspect the listeners first:

```bash
ss -ltnp '( sport = :5566 or sport = :8012 )'
```

If they are stale Televuer/PATH processes from an earlier run, stop the owners
of these dedicated ports and verify they are free:

```bash
sudo fuser -k -TERM 5566/tcp 8012/tcp
sleep 1
ss -ltnp '( sport = :5566 or sport = :8012 )'
```

No listener rows after the `ss` header means the ports are free. `fuser` is
provided by Ubuntu's `psmisc` package. Only stop processes you have identified
as stale Televuer/PATH owners. The launcher checks these ports before startup
and reports occupied port numbers; it does not kill existing processes. Once
the ports are free, rerun `pixi run -e sim mujoco-hand-xr`.

## Local test without Galaxy XR

To check the simulated stand and hand MPPI from an editable JSON pose without
Televuer or PATH, run:

```bash
pixi run -e sim mujoco-hand-test
```

See [the simulation overview](../boost/sim/README.md) for scene and transport
details.
