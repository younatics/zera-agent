from agent.common.slack_notify import notify_slack, send_file_to_slack
import os
import requests

def test_notify_slack():
    webhook_url = "https://hooks.slack.com/services/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"
    notify_slack("Test message: Slack notification is working properly.", webhook_url)
    print("Slack notification test completed.")

def test_send_file_to_slack():
    filepath = os.getenv("SLACK_TEST_FILE", "README.md")
    channel = os.getenv("SLACK_CHANNEL")
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    if not channel or not bot_token:
        print("Please check SLACK_CHANNEL, SLACK_BOT_TOKEN environment variables.")
        return
    send_file_to_slack(filepath, channel, f"Test file upload: {os.path.basename(filepath)}", bot_token)
    print("Slack file upload test completed.")

def test_conversations_open():
    user_id = os.getenv("SLACK_CHANNEL")
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    if not user_id or not bot_token:
        print("Please check SLACK_CHANNEL, SLACK_BOT_TOKEN environment variables.")
        return
    url = "https://slack.com/api/conversations.open"
    headers = {"Authorization": f"Bearer {bot_token}"}
    data = {"users": user_id}
    response = requests.post(url, headers=headers, data=data)
    print("conversations.open response:", response.text)
    if response.ok and response.json().get("ok"):
        channel_id = response.json()["channel"]["id"]
        print(f"DM channel ID: {channel_id}")
    else:
        print("conversations.open failed")

def test_users_info():
    user_id = os.getenv("SLACK_CHANNEL")
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    if not user_id or not bot_token:
        print("Please check SLACK_CHANNEL, SLACK_BOT_TOKEN environment variables.")
        return
    url = "https://slack.com/api/users.info"
    headers = {"Authorization": f"Bearer {bot_token}"}
    params = {"user": user_id}
    response = requests.get(url, headers=headers, params=params)
    print("users.info response:", response.text)
    if response.ok and response.json().get("ok"):
        print(f"User {user_id} exists in the workspace.")
    else:
        print(f"User {user_id} does not exist in the workspace or has insufficient permissions.")

if __name__ == "__main__":
    test_notify_slack()
    test_send_file_to_slack()
    test_conversations_open()
    test_users_info() 