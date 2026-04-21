import os
import tempfile
import unittest

from agent.common.slack_notify import notify_slack, send_file_to_slack


class FakeResponse:
    status_code = 200
    ok = True
    text = "ok"

    def json(self):
        return {"ok": True}


class FakeHttpClient:
    def __init__(self):
        self.calls = []

    def post(self, url, json=None, headers=None, data=None, files=None):
        file_bytes = None
        if files and "file" in files:
            file_bytes = files["file"].read()

        self.calls.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "data": data,
                "file_bytes": file_bytes,
            }
        )
        return FakeResponse()


class SlackNotifyTest(unittest.TestCase):
    def test_notify_slack_posts_message_payload(self):
        client = FakeHttpClient()

        response = notify_slack("hello", "https://example.test/webhook", http_client=client)

        self.assertIsInstance(response, FakeResponse)
        self.assertEqual(client.calls[0]["url"], "https://example.test/webhook")
        self.assertEqual(client.calls[0]["json"], {"text": "hello"})

    def test_send_file_to_slack_posts_upload_payload(self):
        client = FakeHttpClient()
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"content")
            tmp_path = tmp.name

        try:
            response = send_file_to_slack(
                filepath=tmp_path,
                channels="#alerts",
                message="uploaded",
                bot_token="token",
                http_client=client,
            )
        finally:
            os.unlink(tmp_path)

        self.assertIsInstance(response, FakeResponse)
        self.assertEqual(client.calls[0]["url"], "https://slack.com/api/files.upload")
        self.assertEqual(client.calls[0]["headers"], {"Authorization": "Bearer token"})
        self.assertEqual(
            client.calls[0]["data"],
            {"channels": "#alerts", "initial_comment": "uploaded"},
        )
        self.assertEqual(client.calls[0]["file_bytes"], b"content")


if __name__ == "__main__":
    unittest.main()
