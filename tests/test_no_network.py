"""No test may touch the network (tests/conftest.py `_no_network`).

Two NL tests used to send a real Open-Meteo Archive request on every run,
because their fixture stubbed the geocoder and the agent but not the
climatology seam, and passed only because the request failed and the path
degrades quietly. The guard makes such a test fail instead of pass.
"""
import socket

import pytest


def test_a_name_lookup_outside_loopback_is_refused_and_recorded(_no_network):
    with pytest.raises(OSError):
        socket.getaddrinfo("archive-api.open-meteo.com", 443)

    assert _no_network == ["archive-api.open-meteo.com"]
    _no_network.clear()  # recorded attempts fail the test at teardown: this one was the point


def test_a_connect_to_a_public_address_is_refused_and_recorded(_no_network):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError):
            sock.connect(("192.0.2.1", 443))  # TEST-NET-1, never routed
    finally:
        sock.close()

    assert _no_network == ["192.0.2.1"]
    _no_network.clear()


def test_loopback_stays_usable(_no_network):
    assert socket.getaddrinfo("localhost", 80)
    assert _no_network == []
