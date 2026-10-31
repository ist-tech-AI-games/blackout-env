"""
Wire format between the match runner and a team's policy container.

Uses only the standard library and numpy so the policy container can load this file
without installing blackout-env (and its mlagents-envs dependency).

Message = fixed prefix (header_len: u32, body_len: u64, network byte order)
          + JSON header + raw body bytes.

Request  header: {"seq", "agents", "vector_shape": [B, N],
                  "graphic_shape": [U, H, W, C], "graphic_index": [B ints in 0..U-1],
                  "slots": [B ints in 0..4, each agent's position within its team]}
         body  : vector float32[B, N] bytes + graphic uint8[U, H, W, C] bytes
                 (U distinct graphics; agents of one team share one, so usually U = 1)
Response header: {"seq"} or {"seq", "error"}
         body  : action float32[B, 2] bytes (empty on error)
"""

from __future__ import annotations

import json
import socket
import struct

PREFIX = struct.Struct("!IQ")
MAX_HEADER_BYTES = 1 << 16


def encode_msg(header: dict, body: bytes = b"") -> bytes:
    h = json.dumps(header).encode()
    return PREFIX.pack(len(h), len(body)) + h + body


def send_msg(sock: socket.socket, header: dict, body: bytes = b"") -> None:
    sock.sendall(encode_msg(header, body))


def try_parse(buf: bytearray, max_body: int) -> tuple[dict, bytes] | None:
    """
    Pop one complete message from the front of buf.
    Returns None if buf does not yet hold a full message.
    Raises ValueError if the sizes exceed the limits (peer is misbehaving).
    """
    if len(buf) < PREFIX.size:
        return None
    hlen, blen = PREFIX.unpack_from(buf)
    if hlen > MAX_HEADER_BYTES or blen > max_body:
        raise ValueError(f"message too large (header={hlen}, body={blen})")
    end = PREFIX.size + hlen + blen
    if len(buf) < end:
        return None
    header = json.loads(bytes(buf[PREFIX.size:PREFIX.size + hlen]))
    body = bytes(buf[PREFIX.size + hlen:end])
    del buf[:end]
    if not isinstance(header, dict):
        raise ValueError("header is not a JSON object")
    return header, body


def recv_latest(sock: socket.socket, buf: bytearray, max_body: int) -> tuple[dict, bytes]:
    """
    Blocking read of one message, then drop any older requests already queued behind it
    and return the newest one. A slow policy answers the current step instead of
    working through a backlog of stale steps.
    """
    msg = recv_msg(sock, buf, max_body)
    sock.setblocking(False)
    try:
        while True:
            try:
                chunk = sock.recv(1 << 20)
            except BlockingIOError:
                break
            if not chunk:
                raise ConnectionError("peer closed the connection")
            buf += chunk
    finally:
        sock.setblocking(True)
    while (newer := try_parse(buf, max_body)) is not None:
        msg = newer
    return msg


def recv_msg(sock: socket.socket, buf: bytearray, max_body: int) -> tuple[dict, bytes]:
    """Blocking read of one message (used by the policy server)."""
    while True:
        msg = try_parse(buf, max_body)
        if msg is not None:
            return msg
        chunk = sock.recv(1 << 20)
        if not chunk:
            raise ConnectionError("peer closed the connection")
        buf += chunk
