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
    슬랙 채널로 파일을 업로드합니다.
    :param filepath: 업로드할 파일 경로
    :param channels: 업로드할 채널명(예: '#general')
    :param message: 파일과 함께 보낼 메시지
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