import unittest

from ask_vertex import _build_openrouter_reasoning, _sanitize_model_answer


class SanitizeModelAnswerTests(unittest.TestCase):
    def test_strips_leading_think_tags(self):
        text = "<think>reasoning</think>Answer: Paris"
        self.assertEqual(_sanitize_model_answer(text), "Paris")

    def test_strips_leading_reasoning_fence(self):
        text = "```thinking\nstep 1\nstep 2\n```\nFinal Answer: 42"
        self.assertEqual(_sanitize_model_answer(text), "42")

    def test_extracts_final_answer_after_prose_reasoning(self):
        text = "Here is my reasoning:\n1. foo\n2. bar\n\nFinal Answer: Rome"
        self.assertEqual(_sanitize_model_answer(text), "Rome")

    def test_extracts_answer_after_markdown_heading(self):
        text = "## Answer\nSeoul"
        self.assertEqual(_sanitize_model_answer(text), "Seoul")

    def test_unwraps_fenced_plain_answer(self):
        text = "Answer:\n```text\nBerlin\n```"
        self.assertEqual(_sanitize_model_answer(text), "Berlin")

    def test_preserves_plain_non_thinking_response(self):
        text = "The answer is Berlin."
        self.assertEqual(_sanitize_model_answer(text), "The answer is Berlin.")


class OpenRouterReasoningTests(unittest.TestCase):
    def test_disables_openrouter_thinking_by_default(self):
        self.assertEqual(
            _build_openrouter_reasoning(False),
            {
                "effort": "none",
                "exclude": True,
            },
        )

    def test_enables_openrouter_thinking_with_fixed_budget(self):
        self.assertEqual(
            _build_openrouter_reasoning(True),
            {
                "max_tokens": 1024,
                "exclude": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
