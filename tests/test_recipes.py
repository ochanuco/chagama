"""Every registered recipe reaches the worker, the request check, the
catalog and the CLI without being named anywhere else."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from chagama.application.catalog import build_catalog
from chagama.application.generate import validate_request
from chagama.domain.recipes import DEFAULT_RECIPE, RECIPES
from chagama.interfaces import agent, cli

GIT = {"commit": "abc", "branch": "main", "dirty": False}


def request(recipe: str, pose: str) -> dict:
    return {
        "schema_version": 1,
        "request": {"count": 1, "instruction": "test", "seeds": [42]},
        "generation": {"recipe": recipe, "parameters": {"pose": pose}},
        "semantic": {"summary": "test arm"},
    }


def first_pose(name: str) -> str:
    return sorted(RECIPES[name].recipe.poses)[0]


class RecipeRegistryTest(unittest.TestCase):
    def test_default_recipe_is_registered(self):
        self.assertIn(DEFAULT_RECIPE, RECIPES)

    def test_every_pose_has_a_label(self):
        for name, entry in RECIPES.items():
            with self.subTest(recipe=name):
                self.assertEqual(set(entry.recipe.poses), set(entry.pose_labels))

    def test_catalog_lists_exactly_the_registered_recipes(self):
        catalog = build_catalog(GIT)
        self.assertEqual([recipe["name"] for recipe in catalog["recipes"]],
                         list(RECIPES))

    def test_validate_request_accepts_each_registered_recipe(self):
        for name in RECIPES:
            with self.subTest(recipe=name):
                validate_request(request(name, first_pose(name)))

    def test_validate_request_rejects_an_unregistered_recipe_and_lists_them(self):
        with self.assertRaises(SystemExit) as raised:
            validate_request(request("no-such-recipe", "stand"))
        for name in RECIPES:
            self.assertIn(name, str(raised.exception.code))

    def test_worker_builds_and_guards_each_registered_recipe(self):
        for name in RECIPES:
            with self.subTest(recipe=name):
                pose = first_pose(name)
                self.assertIn(name, agent.RECIPES)
                self.assertTrue(agent._identity_tags(name, pose, None))
                self.assertIsNotNone(agent._pose_fingerprint(name, pose))

    def test_cli_prints_each_registered_recipe_prompt(self):
        for name, entry in RECIPES.items():
            pose = first_pose(name)
            with self.subTest(recipe=name):
                output = io.StringIO()
                with patch.object(cli, "ChimeraClient"), redirect_stdout(output):
                    cli.main([name, "prompt", "--pose", pose, "--json"])
                payload = json.loads(output.getvalue())
                self.assertEqual(payload["positive"], entry.recipe.positive(pose))
                self.assertEqual(payload["negative"], entry.recipe.negative(pose))


if __name__ == "__main__":
    unittest.main()
