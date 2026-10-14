from __future__ import annotations

import unittest

import prompt_lint

from chagama.domain.yukari.recipe import negative, positive


class PromptLintScriptTest(unittest.TestCase):
    def test_request_prompts_use_overrides_with_recipe_fallbacks(self):
        request = {"generation": {
            "parameters": {"pose": "coffee", "costume": "roomwear"},
            "prompt": "explicit positive",
        }}
        actual_positive, actual_negative = prompt_lint.request_prompts(request)
        self.assertEqual(actual_positive, "explicit positive")
        self.assertEqual(actual_negative, negative("coffee", "roomwear", None))

        request["generation"]["negative_prompt"] = "explicit negative"
        del request["generation"]["prompt"]
        actual_positive, actual_negative = prompt_lint.request_prompts(request)
        self.assertEqual(actual_positive, positive("coffee", "roomwear", None))
        self.assertEqual(actual_negative, "explicit negative")

    def test_request_prompts_pass_expression_through(self):
        request = {"generation": {
            "parameters": {"pose": "coffee", "expression": "doya"},
        }}
        actual_positive, actual_negative = prompt_lint.request_prompts(request)
        self.assertEqual(actual_positive, positive("coffee", None, "doya"))
        self.assertEqual(actual_negative, negative("coffee", None, "doya"))


if __name__ == "__main__":
    unittest.main()
