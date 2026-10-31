import socket

import pytest

from blackout_env.competition.protocol import encode_msg, recv_pending, try_parse

MAX_BODY = 1 << 20


def test_round_trip():
    buf = bytearray(encode_msg({"seq": 3, "x": [1, 2]}, b"\x00\x01\x02"))
    assert try_parse(buf, MAX_BODY) == ({"seq": 3, "x": [1, 2]}, b"\x00\x01\x02")
    assert buf == b""


def test_partial_message_waits_for_more():
    msg = encode_msg({"seq": 1}, b"abcdef")
    buf = bytearray(msg[:-2])
    assert try_parse(buf, MAX_BODY) is None
    buf += msg[-2:]
    assert try_parse(buf, MAX_BODY) == ({"seq": 1}, b"abcdef")


def test_two_messages_in_one_buffer():
    buf = bytearray(encode_msg({"seq": 1}) + encode_msg({"seq": 2}, b"z"))
    assert try_parse(buf, MAX_BODY)[0]["seq"] == 1
    assert try_parse(buf, MAX_BODY) == ({"seq": 2}, b"z")


def test_oversized_body_is_rejected():
    buf = bytearray(encode_msg({"seq": 1}, b"x" * 100))
    with pytest.raises(ValueError):
        try_parse(buf, max_body=10)


def test_header_must_be_an_object():
    buf = bytearray(encode_msg([1, 2]))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        try_parse(buf, MAX_BODY)


def test_recv_pending_returns_queued_requests_oldest_first():
    a, b = socket.socketpair()
    try:
        for seq in (1, 2, 3):
            a.sendall(encode_msg({"seq": seq}))
        msgs = recv_pending(b, bytearray(), MAX_BODY)
        assert [h["seq"] for h, _ in msgs] == [1, 2, 3]
    finally:
        a.close()
        b.close()
