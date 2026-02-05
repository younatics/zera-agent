import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.common.slack_notify import notify_slack, send_file_to_slack


class DummyResponse:
    def __init__(self, *, status_code=200, ok=True, text="ok", json_data=None):
        self.status_code = status_code
        self.ok = ok
        self.text = text
        self._json_data = json_data or {}

    def json(self):
        return self._json_data


def test_notify_slack_success(monkeypatch):
    recorded = {}

    def fake_post(url, *, json=None, timeout=None):
        recorded["url"] = url
        recorded["json"] = json
        recorded["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr("agent.common.slack_notify.requests.post", fake_post)

    assert notify_slack("Ping", "https://example.com/webhook") is True
    assert recorded["json"] == {"text": "Ping"}
    assert recorded["timeout"] == 5.0


def test_notify_slack_failure_logs(monkeypatch, capsys):
    def fake_post(url, *, json=None, timeout=None):
        return DummyResponse(status_code=500, ok=False, text="error")

    monkeypatch.setattr("agent.common.slack_notify.requests.post", fake_post)

    assert notify_slack("Ping", "https://example.com/webhook") is False
    captured = capsys.readouterr().out
    assert "Slack notification failed" in captured


def test_send_file_to_slack_success(monkeypatch, tmp_path):
    recorded = {}

    def fake_post(url, *, headers=None, data=None, files=None, timeout=None):
        recorded["url"] = url
        recorded["headers"] = headers
        recorded["data"] = data
        recorded["timeout"] = timeout
        recorded["file_bytes"] = files["file"].read()
        return DummyResponse(json_data={"ok": True})

    monkeypatch.setattr("agent.common.slack_notify.requests.post", fake_post)

    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello")

    assert send_file_to_slack(str(file_path), "#general", "A file", "token") is True
    assert recorded["headers"]["Authorization"] == "Bearer token"
    assert recorded["data"]["channels"] == "#general"
    assert recorded["file_bytes"] == b"hello"
    assert recorded["timeout"] == 10.0


def test_send_file_to_slack_handles_exception(monkeypatch, tmp_path, capsys):
    def fake_post(*args, **kwargs):
        raise requests.RequestException("boom")

    monkeypatch.setattr("agent.common.slack_notify.requests.post", fake_post)

    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello")

    assert send_file_to_slack(str(file_path), "#general", "A file", "token") is False
    assert "Slack file upload failed" in capsys.readouterr().out
