from __future__ import annotations

from typing import Any


def _load_requests():
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise ImportError(
            "The 'requests' package is required to send Slack notifications. "
            "Install project dependencies before calling Slack helpers."
        ) from exc
    return requests


def notify_slack(message: str, webhook_url: str, http_client: Any = None):
    client = http_client or _load_requests()
    response = client.post(webhook_url, json={"text": message})
    if response.status_code != 200:
        print(f"Slack notification failed: {response.text}")
    return response


def send_file_to_slack(
    filepath: str,
    channels: str,
    message: str,
    bot_token: str,
    http_client: Any = None,
):
    client = http_client or _load_requests()
    with open(filepath, "rb") as file_content:
        response = client.post(
            "https://slack.com/api/files.upload",
            headers={"Authorization": f"Bearer {bot_token}"},
            data={"channels": channels, "initial_comment": message},
            files={"file": file_content},
        )

    if not response.ok or not response.json().get("ok", False):
        print(f"Slack file upload failed: {response.text}")
    return response
