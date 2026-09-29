"""Strict ASCII GTIN validation shared by Gold identity gates.

Canonical GTIN values are GS1 identifiers, not arbitrary Unicode digit strings.
Formatting characters are intentionally not normalized here: callers must retain
the original identifier as unjoinable unless it is already an exact GTIN value.
"""
from __future__ import annotations

import re


GTIN_LENGTHS = frozenset({8, 12, 13, 14})
_ASCII_GTIN = re.compile(r"^[0-9]+$")


def is_valid_gtin(value: object) -> bool:
    """Return true only for an ASCII GTIN-8/-12/-13/-14 with a valid check digit."""
    text = str(value or "")
    if len(text) not in GTIN_LENGTHS or _ASCII_GTIN.fullmatch(text) is None:
        return False
    expected = (10 - sum(
        int(digit) * (3 if (len(text) - index) % 2 == 0 else 1)
        for index, digit in enumerate(text[:-1])
    ) % 10) % 10
    return int(text[-1]) == expected
