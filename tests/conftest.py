"""Shared test setup: every test runs offline.

Unit tests must never touch the network. This fixture forces the Hugging Face
libraries offline and makes any socket connection attempt fail loudly.
"""

import socket

import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def refuse(self, *args, **kwargs):
        raise AssertionError("network access attempted during a test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
