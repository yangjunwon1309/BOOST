"""teleimager's ``ImageClient`` over cameras that live on more than one board.

Every camera used to hang off one machine, so one address and teleimager's own
``ImageClient`` covered the whole rig. The right wrist is now wired into the G1
PC2 while the head and left wrist stay on Thor, because Thor's USB could not
hold all three. Each camera therefore carries its own host here, and each host
is asked only for the cameras it owns. Ports are whatever that server reports,
so both servers can keep the stock 55555/55556/55557.

With one address for all three cameras this is the same single config request
and the same subscriptions as before.

This mirrors ``url_groot_deploy/client/camera_source.py`` so that recording and
deployment resolve the camera hosts the same way.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import zmq

import logging_mp
logger_mp = logging_mp.getLogger(__name__)


CAMERA_NAMES = ("head_camera", "left_wrist_camera", "right_wrist_camera")

# Short spellings, the same ones url_groot_deploy accepts.
_CAMERA_ALIASES = {
    "head": "head_camera",
    "left_wrist": "left_wrist_camera",
    "right_wrist": "right_wrist_camera",
}


def resolve_camera_hosts(default_host: str,
                         overrides: Optional[Mapping[str, str]] = None) -> dict:
    """Give every camera an address, falling back to the rig-wide one.

    An unknown camera key is an error: a typo would otherwise be ignored and
    leave the camera pointing at the wrong board.
    """
    resolved = {name: str(default_host) for name in CAMERA_NAMES}
    for key, host in dict(overrides or {}).items():
        name = _CAMERA_ALIASES.get(str(key), str(key))
        if name not in resolved:
            raise ValueError(f"unknown camera {key!r} (expected one of "
                             f"{', '.join(CAMERA_NAMES)} or {', '.join(_CAMERA_ALIASES)})")
        if host is None or not str(host).strip():
            continue
        resolved[name] = str(host).strip()
    return resolved


def parse_camera_host_pairs(pairs) -> dict:
    """Turn repeated ``--camera-host-for CAMERA=HOST`` into a mapping."""
    merged = {}
    for pair in list(pairs or []):
        camera, separator, host = str(pair).partition("=")
        if not separator or not camera.strip() or not host.strip():
            raise ValueError(f"--camera-host-for expects CAMERA=HOST, got {pair!r}")
        merged[camera.strip()] = host.strip()
    # Reject an unknown camera at startup rather than at the first frame.
    resolve_camera_hosts("unused", merged)
    return merged


def request_camera_config(host: str, request_port: int = 60000, timeout_ms: int = 2000) -> dict:
    """Ask one teleimager server for its camera config.

    teleimager's ``ZMQ_Requester`` writes the reply over the package-wide
    ``cam_config_client.yaml`` and falls back to that file when the server does
    not answer. With the cameras split over two servers each reply covers only
    part of the rig, so that file would hold whichever half was asked last and a
    dead server would look healthy. Speak the protocol directly and fail loudly.
    """
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(f"tcp://{host}:{int(request_port)}")
        socket.send(b"GET_DATA")
        if not socket.poll(int(timeout_ms), zmq.POLLIN):
            raise RuntimeError(f"no teleimager camera config from {host}:{request_port} within "
                               f"{timeout_ms} ms -- is teleimager-server running on that host?")
        config = socket.recv_json()
    finally:
        socket.close(linger=0)
        context.term()
    if not isinstance(config, dict):
        raise RuntimeError(f"teleimager at {host}:{request_port} returned an unusable config")
    return config


def merge_camera_configs(hosts: Mapping[str, str], remote_configs: Mapping[str, dict]) -> dict:
    """Take each camera's section from the server it is wired into.

    A server that does not own a camera still carries a placeholder section for
    it with every stream disabled, so the other server's section is never
    consulted for that camera.
    """
    merged = {}
    for name in CAMERA_NAMES:
        host = hosts[name]
        section = remote_configs[host].get(name)
        if not isinstance(section, dict):
            raise RuntimeError(f"teleimager at {host} does not describe {name}; "
                               "point --camera-host-for at the board it is wired into")
        merged[name] = section
    return merged


class MultiHostImageClient:
    """The ``ImageClient`` surface teleop_ikea.py uses, over per-camera hosts."""

    def __init__(self, hosts: Mapping[str, str], request_port: int = 60000,
                 request_bgr: bool = False, request_timeout_ms: int = 2000,
                 start_legacy_subscriptions: bool = True):
        self._hosts = dict(hosts)
        self._request_bgr = bool(request_bgr)
        self._closed = False
        self._subscriber_manager = None
        self._legacy_started = False

        remote = {host: request_camera_config(host, request_port, request_timeout_ms)
                  for host in sorted(set(self._hosts.values()))}
        self._cam_config = merge_camera_configs(self._hosts, remote)

        if start_legacy_subscriptions:
            self.start_legacy_subscriptions()

        head = self._cam_config["head_camera"]
        if not head.get("enable_zmq", False) and not head.get("enable_webrtc", False):
            logger_mp.warning("[MultiHostImageClient] head camera is enabled on neither ZMQ nor WebRTC.")

    def start_legacy_subscriptions(self) -> None:
        """Start legacy images once, after choosing them over TIM2 recording.

        A config-only client does not create subscriber threads or decode BGR.
        Existing callers retain eager subscriptions through the default flag.
        """
        if self._closed:
            raise RuntimeError("cannot start legacy subscriptions on a closed image client")
        if self._legacy_started:
            return
        from teleimager.image_client import ZMQ_SubscriberManager

        manager = ZMQ_SubscriberManager.get_instance()
        try:
            for name, section in self._cam_config.items():
                if section.get("enable_zmq", False):
                    manager.subscribe(self._hosts[name], section["zmq_port"],
                                      request_bgr=self._request_bgr)
        except BaseException:
            manager.close()
            raise
        self._subscriber_manager = manager
        self._legacy_started = True

    def host_for(self, name: str) -> str:
        return self._hosts[name]

    def _frame(self, name: str) -> Any:
        section = self._cam_config[name]
        if not section.get("enable_zmq", False) or self._subscriber_manager is None:
            return None
        return self._subscriber_manager.subscribe(self._hosts[name], section["zmq_port"],
                                                  request_bgr=self._request_bgr)

    def get_cam_config(self) -> dict:
        return self._cam_config

    def get_head_frame(self):
        return self._frame("head_camera")

    def get_left_wrist_frame(self):
        return self._frame("left_wrist_camera")

    def get_right_wrist_frame(self):
        return self._frame("right_wrist_camera")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._subscriber_manager is not None:
            self._subscriber_manager.close()
        logger_mp.info("[MultiHostImageClient] closed.")
