# akari

> Kizuna Akari belongs to her original creators and rights holders -- see
> [Derivative work](../../README.md#derivative-work) in the README.

The second recipe on the Anima style. She draws the same poses and
expressions as [yukari](../yukari/anima.md) by name (`POSES` and
`EXPRESSIONS` under `domain/anima/`); only what is hers lives in
`src/chagama/domain/akari/`. The delivery identity is yukari's except the
outer band and the backdrop colours: a deliver of an akari picture that
names no `outlines` draws the white band, then a `#c27138` rim, and her
backdrop patterns are painted
on peach `#f7d3b5` deepened toward that rim (`delivery.py`). `reference`
builds onto `yukari_persona` only.

```bash
uv run chagama akari prompt --pose bust
```

## What is hers

| File | Holds |
|---|---|
| `prompt_style.py` | `kizuna akari`, identity (white twin braids, deep blue eyes with pale pink pupils), the droopy eye shape, `EYES` (her lower-lid droop, the eye quality under every expression), face, body, the `1990s (style)` suffix on the style block, her negative bans (eye shape, purple/black tights, black jacket, accessories, mechanical body) |
| `costumes.py` | `half_jacket`, her only costume and the default: the very dark brown jacket slipping off the shoulders over a dark grey pinafore with a white frilled, lace-trimmed hem; a from-behind variant (hem included); white tights with a faint light-orange tint for every legwear kind |
| `looks.py` | her looks on the shared poses: the adult body of `bust` / `gao` / `avatar`, the bare face of the `nomakeup` poses |
| `delivery.py` | her default deliver bands (yukari's white band, then the `#c27138` rim) and backdrop palette (base `#f7d3b5`, yukari's pink accent, deep stroke `#c27138`) |
| `character.py` / `recipe.py` | `AKARI` and `RECIPE`, registered as `akari` in `domain/recipes.py`; persona pose `akari_persona` |

## Contracts

- Every shared pose wears `half_jacket` unless a request names a costume.
- The identity guard holds her name, hair, eye colour and `tareme` /
  `droopy eyes` (`tests/test_akari.py`).
- None of yukari's words (name, purple, ジト目, hood, hair ornament) and
  none of her bans reach an akari prompt; the test checks every pose and
  expression.

## Does not work

- An orange or peach legwear colour (`light orange tint` at 1.3,
  `pale peach tights`, `peach pantyhose`): Anima Turbo draws her official
  orange vertical-striped tights and orange arm warmers, and
  `plain pantyhose` / `solid color legwear` with striped bans in the
  negative do not stop it. Ivory / cream, or skin showing through the
  white, is the wording tried instead. `[Anima]`
