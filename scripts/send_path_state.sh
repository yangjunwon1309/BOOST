#!/usr/bin/env bash
set -euo pipefail
state="${1:?usage: $0 teleop|intention|auto_planning|damped_return}"
port="${PATH_STATE_COMMAND_PORT:-5563}"
topic="${PATH_STATE_COMMAND_TOPIC:-path_state_command}"
PATH_STATE_TO_SEND="$state" python3 - <<'INNERPY'
import msgpack, os, time, zmq
state = os.environ["PATH_STATE_TO_SEND"]
port = os.environ.get("PATH_STATE_COMMAND_PORT", "5563")
topic = os.environ.get("PATH_STATE_COMMAND_TOPIC", "path_state_command")
ctx = zmq.Context.instance()
s = ctx.socket(zmq.PUB)
s.connect(f"tcp://127.0.0.1:{port}")
time.sleep(0.2)
payload = {"schema": "path.state_command.v1", "state": state, "timestamp_us": time.monotonic_ns() // 1000}
s.send(topic.encode() + msgpack.packb(payload, use_bin_type=True))
print("sent", payload)
INNERPY
