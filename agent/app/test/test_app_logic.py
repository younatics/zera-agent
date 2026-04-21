import unittest

from agent.app.app_logic import build_test_cases


class AppLogicTest(unittest.TestCase):
    def test_build_test_cases_for_csv_records(self):
        test_cases, display_data = build_test_cases(
            [{"question": "q", "expected_answer": "a"}],
            "CSV",
        )

        self.assertEqual(test_cases, [{"question": "q", "expected": "a"}])
        self.assertEqual(display_data, [{"question": "q", "expected_answer": "a"}])

    def test_build_test_cases_for_multiple_choice_dataset(self):
        test_cases, _ = build_test_cases(
            [
                {
                    "question": "Pick one",
                    "choices": ["first", "second"],
                    "answer": 1,
                }
            ],
            "MMLU",
        )

        self.assertEqual(test_cases[0]["expected"], "B")
        self.assertIn("A. first", test_cases[0]["question"])
        self.assertIn("B. second", test_cases[0]["question"])

    def test_build_test_cases_normalizes_cnn_summary_bullets(self):
        test_cases, _ = build_test_cases(
            [{"input": "article", "expected_answer": "- bullet\nfinal summary"}],
            "CNN",
        )

        self.assertEqual(test_cases[0]["expected"], "final summary")

    def test_build_test_cases_rejects_unknown_dataset(self):
        with self.assertRaises(ValueError):
            build_test_cases([{"question": "q"}], "Unknown")


if __name__ == "__main__":
    unittest.main()
