import requests

def notify_slack(message: str, webhook_url: str):
    payload = {
        "text": message
    }
    response = requests.post(webhook_url, json=payload)
    if response.status_code != 200:
        print(f"Slack notification failed: {response.text}")

def send_file_to_slack(filepath: str, channels: str, message: str, bot_token: str):
    """
    Upload a file to a Slack channel.
    :param filepath: Path to the file to upload
    :param channels: Channel name to upload to (e.g., '#general')
    :param message: Message to send with the file
    :param bot_token: Bot User OAuth Token
    """
    with open(filepath, "rb") as file_content:
        response = requests.post(
            "https://slack.com/api/files.upload",
            headers={
                "Authorization": f"Bearer {bot_token}"
            },
            data={
                "channels": channels,
                "initial_comment": message
            },
            files={"file": file_content}
        )
    if not response.ok or not response.json().get("ok", False):
        print(f"Slack file upload failed: {response.text}") 