"""Component-model invariants: Section/Priority ordering, the GENERAL-section
priority table `_components` assigns, part-mapping coverage, and every
pose's Framing."""

from __future__ import annotations

import unittest

from chagama.domain.yukari.components import (
    PART_OF,
    Component,
    Framing,
    Priority,
    Section,
    assemble,
)
from chagama.domain.yukari.framing import FRAMING
from chagama.domain.yukari.poses import POSES
from chagama.domain.yukari.recipe import (
    PART_GROUPS,
    PART_NAMES,
    _components,
    positive,
    positive_parts,
    render_spec,
)


class AssembleOrderTest(unittest.TestCase):
    def test_sorts_by_section_then_priority(self):
        components = (
            Component("b", Section.GENERAL, Priority.TAIL, "b"),
            Component("a", Section.COUNT, Priority.MAIN, "a"),
            Component("c", Section.GENERAL, Priority.LEAD, "c"),
            Component("d", Section.QUALITY, Priority.MAIN, "d"),
            Component("e", Section.GENERAL, Priority.LEAD, "e"),
        )
        self.assertEqual([c.name for c in assemble(components)],
                         ["d", "a", "c", "e", "b"])

    def test_equal_keys_keep_declaration_order(self):
        components = (
            Component("first", Section.GENERAL, Priority.MAIN, "1"),
            Component("second", Section.GENERAL, Priority.MAIN, "2"),
            Component("third", Section.GENERAL, Priority.MAIN, "3"),
        )
        self.assertEqual([c.name for c in assemble(components)],
                         ["first", "second", "third"])


class GeneralPriorityTableTest(unittest.TestCase):
    LEAD = frozenset({"identity", "eye_base", "eye_quality", "framing_tags",
                      "body_build", "leg_display"})
    MAIN = frozenset({"action", "mouth", "mood", "gesture", "nails", "costume",
                      "legwear", "cutout"})
    TAIL = frozenset({"face", "style"})

    def test_general_components_match_the_priority_table(self):
        for pose in POSES:
            with self.subTest(pose=pose):
                for component in _components(pose):
                    if component.section != Section.GENERAL:
                        continue
                    if component.name in self.LEAD:
                        expected = Priority.LEAD
                    elif component.name in self.MAIN:
                        expected = Priority.MAIN
                    elif component.name in self.TAIL:
                        expected = Priority.TAIL
                    else:
                        self.fail(f"{component.name} not in the priority table")
                    self.assertEqual(component.priority, expected,
                                     component.name)


class PartMappingTest(unittest.TestCase):
    def test_every_part_name_is_covered(self):
        self.assertEqual(set(PART_OF.values()), set(PART_NAMES))

    def test_every_declared_component_maps_to_a_known_part(self):
        for pose in POSES:
            for component in _components(pose):
                self.assertIn(component.name, PART_OF, component.name)

    def test_every_legacy_groups_members_exist_in_positive_parts(self):
        # The priority table interleaves a legacy part's members with other
        # parts' components, so `patches.py` resolves a legacy
        # `prompt.positive.<part>` patch against each member individually
        # rather than a contiguous run; every member still has to be an
        # assembled component.
        for pose in POSES:
            with self.subTest(pose=pose):
                names = {name for name, _ in positive_parts(pose)}
                for part, members in PART_GROUPS.items():
                    for member in members:
                        self.assertIn(member, names, f"{part}.{member}")


class PriorityDrivenOrderTest(unittest.TestCase):
    def test_lead_general_component_sorts_right_after_non_general_sections(self):
        G, M, L = Section.GENERAL, Priority.MAIN, Priority.LEAD
        declared = (
            Component("quality", Section.QUALITY, M, "q"),
            Component("artist", Section.ARTIST, M, "ar"),
            Component("action", G, M, "ac"),
            Component("mood", G, L, "mo"),
            Component("gesture", G, M, "ge"),
        )
        sorted_components = assemble(declared)
        names = [c.name for c in sorted_components]
        self.assertEqual(names, ["quality", "artist", "mood", "action", "gesture"])

        # positive_parts() builds (name, text) directly from `assemble`'s
        # order, so the joined prompt matches that order exactly.
        parts = tuple((c.name, c.text) for c in sorted_components)
        self.assertEqual([name for name, _ in parts], names)
        self.assertEqual(
            "".join(text for _, text in parts),
            "".join(c.text for c in sorted_components))


class FramingTest(unittest.TestCase):
    def test_every_pose_has_a_framing(self):
        for pose, spec in POSES.items():
            with self.subTest(pose=pose):
                self.assertIsInstance(spec.framing, Framing)

    def test_framing_assignments(self):
        expected = {
            "cinema": Framing.COWBOY, "coffee": Framing.COWBOY,
            "amae": Framing.COWBOY, "gao": Framing.UPPER,
            "step": Framing.FULL, "stand": Framing.FULL, "dance": Framing.FULL,
            "bust": Framing.BUST, "anyo": Framing.FULL,
            "pillow": Framing.CLOSE,
        }
        self.assertEqual({pose: spec.framing for pose, spec in POSES.items()},
                         expected)

    def test_angle_and_leg_display_are_disjoint_strings(self):
        for pose, spec in POSES.items():
            with self.subTest(pose=pose):
                self.assertIsInstance(spec.angle, str)
                self.assertIsInstance(spec.leg_display, str)


class FramingTagsComponentTest(unittest.TestCase):
    def test_framing_tags_component_is_angle_plus_framing_text(self):
        for pose, spec in POSES.items():
            with self.subTest(pose=pose):
                parts = dict(positive_parts(pose))
                self.assertEqual(
                    parts["framing_tags"],
                    spec.angle + FRAMING[spec.framing].text)

    def test_every_framing_kind_has_a_spec(self):
        for framing in Framing:
            with self.subTest(framing=framing):
                self.assertIn(framing, FRAMING)
                self.assertIsInstance(FRAMING[framing].text, str)

    def test_framing_texts(self):
        self.assertEqual(FRAMING[Framing.BUST].text,
                         "(portrait:1.5), (head and shoulders:1.4), "
                         "(upper body:1.35), (face focus:1.3), ")
        self.assertEqual(FRAMING[Framing.COWBOY].text, "(cowboy shot:1.3), ")
        self.assertEqual(FRAMING[Framing.FULL].text,
                         "(full body:1.45), (wide shot:1.3), ")
        self.assertEqual(FRAMING[Framing.UPPER].text, "(upper body:1.3), ")
        self.assertEqual(FRAMING[Framing.LYING].text,
                         "(lying:1.3), (full body:1.35), ")
        self.assertEqual(FRAMING[Framing.CLOSE].text,
                         "(close-up:1.15), (cropped:1.2), ")

    def test_only_bust_framing_carries_a_canvas(self):
        self.assertEqual(FRAMING[Framing.BUST].canvas, (1280, 2048))
        for framing in (Framing.COWBOY, Framing.FULL, Framing.UPPER,
                        Framing.LYING, Framing.CLOSE):
            with self.subTest(framing=framing):
                self.assertIsNone(FRAMING[framing].canvas)


class CanvasResolutionTest(unittest.TestCase):
    def test_pose_canvas_wins_over_framing_canvas(self):
        spec = render_spec("bust", 1, "p")
        self.assertEqual((spec.width, spec.height), (1280, 2048))

    def test_framing_canvas_applies_when_pose_has_none(self):
        # `bust` carries no pose-level canvas of its own; the (1280, 2048)
        # canvas comes from its BUST framing.
        self.assertIsNone(POSES["bust"].canvas)

    def test_non_bust_pose_falls_back_to_the_recipe_default_canvas(self):
        spec = render_spec("stand", 1, "p")
        self.assertEqual((spec.width, spec.height), (1024, 1640))


class NoPlaceTagsTest(unittest.TestCase):
    """Cut-out delivery needs an empty green background; a location tag
    invites furniture or a backdrop the matte then has to cut around."""

    LOCATION_WORDS = ("street", "outdoors", "movie theater", "theater lobby",
                      "indoors", "carpet", "cobblestone", "stone floor",
                      "day", "shopping", "dim lighting")

    def test_no_place_component(self):
        for pose in POSES:
            with self.subTest(pose=pose):
                names = {name for name, _ in positive_parts(pose)}
                self.assertNotIn("place", names)

    def test_no_pose_carries_a_location_tag(self):
        for pose in POSES:
            text = positive(pose)
            for word in self.LOCATION_WORDS:
                with self.subTest(pose=pose, word=word):
                    self.assertNotIn(word, text)


if __name__ == "__main__":
    unittest.main()
