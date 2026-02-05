import requests


def notify_slack(message: str, webhook_url: str, *, timeout: float = 5.0) -> bool:
    """Send a simple Slack notification via webhook.

    Returns ``True`` when Slack acknowledges the request, otherwise ``False``.
    ``requests`` level exceptions are caught to avoid crashing the caller during tests.
    """

    payload = {"text": message}
    try:
        response = requests.post(webhook_url, json=payload, timeout=timeout)
        if response.status_code != 200:
            print(f"Slack notification failed: {response.text}")
            return False
        return True
    except requests.RequestException as exc:
        print(f"Slack notification request failed: {exc}")
        return False


def send_file_to_slack(
    filepath: str,
    channels: str,
    message: str,
    bot_token: str,
    *,
    timeout: float = 10.0,
) -> bool:
    """Upload a file to Slack and return whether the request succeeded."""

    with open(filepath, "rb") as file_content:
        try:
            response = requests.post(
                "https://slack.com/api/files.upload",
                headers={"Authorization": f"Bearer {bot_token}"},
                data={"channels": channels, "initial_comment": message},
                files={"file": file_content},
                timeout=timeout,
            )
        except requests.RequestException as exc:
            print(f"Slack file upload failed: {exc}")
            return False

    ok = response.ok and bool(response.json().get("ok", False))
    if not ok:
        print(f"Slack file upload failed: {response.text}")
    return ok