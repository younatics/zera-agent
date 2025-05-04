from agent.common.slack_notify import notify_slack, send_file_to_slack
import os
import requests

def test_notify_slack():
    webhook_url = "https://hooks.slack.com/services/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"
    notify_slack("테스트 메시지: 슬랙 알림이 정상적으로 동작합니다.", webhook_url)
    print("슬랙 알림 테스트 완료.")

def test_send_file_to_slack():
    filepath = os.getenv("SLACK_TEST_FILE", "README.md")
    channel = os.getenv("SLACK_CHANNEL")
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    if not channel or not bot_token:
        print("SLACK_CHANNEL, SLACK_BOT_TOKEN 환경변수를 확인하세요.")
        return
    send_file_to_slack(filepath, channel, f"테스트 파일 업로드: {os.path.basename(filepath)}", bot_token)
    print("슬랙 파일 업로드 테스트 완료.")

def test_conversations_open():
    user_id = os.getenv("SLACK_CHANNEL")
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    if not user_id or not bot_token:
        print("SLACK_CHANNEL, SLACK_BOT_TOKEN 환경변수를 확인하세요.")
        return
    url = "https://slack.com/api/conversations.open"
    headers = {"Authorization": f"Bearer {bot_token}"}
    data = {"users": user_id}
    response = requests.post(url, headers=headers, data=data)
    print("conversations.open 응답:", response.text)
    if response.ok and response.json().get("ok"):
        channel_id = response.json()["channel"]["id"]
        print(f"DM 채널 ID: {channel_id}")
    else:
        print("conversations.open 실패")

def test_users_info():
    user_id = os.getenv("SLACK_CHANNEL")
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    if not user_id or not bot_token:
        print("SLACK_CHANNEL, SLACK_BOT_TOKEN 환경변수를 확인하세요.")
        return
    url = "https://slack.com/api/users.info"
    headers = {"Authorization": f"Bearer {bot_token}"}
    params = {"user": user_id}
    response = requests.get(url, headers=headers, params=params)
    print("users.info 응답:", response.text)
    if response.ok and response.json().get("ok"):
        print(f"사용자 {user_id}는 워크스페이스에 존재합니다.")
    else:
        print(f"사용자 {user_id}는 워크스페이스에 존재하지 않거나, 권한이 부족합니다.")

if __name__ == "__main__":
    test_notify_slack()
    test_send_file_to_slack()
    test_conversations_open()
    test_users_info() 