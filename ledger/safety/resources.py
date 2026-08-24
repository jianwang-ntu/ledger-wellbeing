"""Crisis resources, kept as data so the routing logic stays testable offline.

Every entry here is a publicly published helpline number. Nothing in this file
is generated, inferred, or personalised: the router picks a row, it never
composes advice.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Helpline:
    region: str
    name: str
    contact: str
    hours: str


# Kept deliberately short. A line is listed only when its number is published by
# the operating organisation itself; anything unverified is omitted rather than
# guessed, because a wrong crisis number is worse than no number.
HELPLINES: tuple[Helpline, ...] = (
    Helpline("SG", "Samaritans of Singapore (SOS)", "1767", "24 hours"),
    Helpline("SG", "SOS Care Text", "https://www.sos.org.sg/", "24 hours"),
    Helpline("US", "988 Suicide & Crisis Lifeline", "988", "24 hours"),
    Helpline("GB", "Samaritans", "116 123", "24 hours"),
    Helpline("INTL", "Find A Helpline (directory, 130+ countries)",
             "https://findahelpline.com/", "directory"),
)

#: Shown when no region is known. The directory is region-agnostic by design.
FALLBACK_REGION = "INTL"


def for_region(region: str | None) -> tuple[Helpline, ...]:
    """Return the helplines for `region`, always including the directory.

    An unknown or absent region is not an error and never yields an empty list —
    the crisis path must not be able to fail closed into showing nothing.
    """
    key = (region or "").strip().upper()
    local = tuple(h for h in HELPLINES if h.region == key)
    directory = tuple(h for h in HELPLINES if h.region == FALLBACK_REGION)
    return local + directory
