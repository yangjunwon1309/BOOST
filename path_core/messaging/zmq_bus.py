from __future__ import annotations

import msgpack
import zmq


def pack_topic(topic: str, payload: dict) -> bytes:
    return topic.encode() + msgpack.packb(payload, use_bin_type=True)


def unpack_topic(raw: bytes, topic: str) -> dict:
    prefix = topic.encode()
    if not raw.startswith(prefix):
        raise ValueError(f"Unexpected topic, expected {topic!r}")
    return msgpack.unpackb(raw[len(prefix):], raw=False)


def make_pub(bind_endpoint: str) -> zmq.Socket:
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.PUB)
    socket.bind(bind_endpoint)
    return socket


def make_sub(connect_endpoint: str, topic: str, conflate: bool = True) -> zmq.Socket:
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.SUB)
    if conflate:
        socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.SUBSCRIBE, topic.encode())
    socket.connect(connect_endpoint)
    return socket

