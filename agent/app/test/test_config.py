import os
import unittest
from unittest.mock import patch

from agent.app.config import missing_api_keys


class AppConfigTest(unittest.TestCase):
    def test_missing_api_keys_ignores_local_models(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(missing_api_keys(["local1", "local2"]), [])

    def test_missing_api_keys_reports_remote_model_keys(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                missing_api_keys(["solar", "gpt4o"]),
                ["OPENAI_API_KEY", "SOLAR_API_KEY"],
            )

    def test_missing_api_keys_deduplicates_models(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                missing_api_keys(["gpt4o", "gpt4o"]),
                ["OPENAI_API_KEY"],
            )


if __name__ == "__main__":
    unittest.main()
