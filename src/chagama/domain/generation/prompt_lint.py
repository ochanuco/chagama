"""Pure positive/negative prompt contradiction checks."""

from __future__ import annotations

import re


COLORS = {"black", "white", "grey", "gray", "blue", "aqua", "navy", "brown",
          "green", "pink", "purple", "red", "yellow", "orange", "charcoal",
          "lavender", "periwinkle", "cream", "silver"}
SYNONYM = {"see-through": "sheer", "transparent": "sheer", "sheer": "sheer",
           "aqua": "blue", "navy": "blue", "periwinkle": "blue",
           "gray": "grey", "charcoal": "grey", "glossy": "shiny",
           "wet": "shiny"}
PROPERTIES = COLORS | {"sheer", "opaque", "shiny", "ribbed", "knit", "muted"}
LEGWEAR = {"legwear", "pantyhose", "thighhighs", "thighhigh", "kneehighs",
           "socks", "tights", "stockings"}
UNSCOPED = {"clothing", "clothes", "tint", "look", "color", "colors"}


def tags(prompt: str) -> list[str]:
    result = []
    for part in prompt.split(","):
        part = re.sub(r":\d+(\.\d+)?$", "", part.strip().strip("()"))
        if part:
            result.append(part)
    return result


def properties(tag: str) -> set[str]:
    words = set(tag.replace("-", " ").split()) | set(tag.split())
    known = {SYNONYM.get(prop, prop) for prop in PROPERTIES}
    return {SYNONYM.get(word, word) for word in words
            if SYNONYM.get(word, word) in known}


def nouns(tag: str) -> set[str]:
    words = set(tag.split())
    result = set()
    if words & LEGWEAR:
        result.add("legwear")
    if "background" in words:
        result.add("background")
    if words & UNSCOPED:
        result.add("*")
    rest = [word for word in tag.split()
            if word not in PROPERTIES and SYNONYM.get(word, word) not in PROPERTIES]
    if rest:
        result.add(rest[-1])
    if not result:
        result.add("*")
    return result


def conflicts(positive: str, negative: str) -> list[tuple[str, str, str]]:
    hits = []
    negative_tags = [(tag, properties(tag), nouns(tag)) for tag in tags(negative)]
    for positive_tag in tags(positive):
        positive_properties = properties(positive_tag)
        positive_nouns = nouns(positive_tag)
        if not positive_properties:
            continue
        for negative_tag, negative_properties, negative_nouns in negative_tags:
            shared = positive_properties & negative_properties
            if not shared:
                continue
            if ("sheer" in positive_properties
                    and ({"solid", "opaque"} & set(negative_tag.split()))):
                continue
            if ("*" in positive_nouns or "*" in negative_nouns
                    or positive_nouns & negative_nouns):
                hits.append((positive_tag, negative_tag, "/".join(sorted(shared))))
    return hits
