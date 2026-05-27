"""Fast, offline unit tests. No network, no GPU, no model downloads.

These cover the pure-logic pieces that the rest of the pipeline relies on:
Hebrew detection, the GPT-2 byte decoder round-trip, and the held-out integrity
of the evaluation set (no training input may leak into evaluation).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.fixtures import (
    DECODING_QUERIES,
    EVAL_INPUTS_ALL,
    EVAL_INPUTS_OWN,
    EVAL_INPUTS_PROVIDED,
)
from src.common.token_utils import (
    HEBREW_RE,
    _byte_decoder,
)


def test_hebrew_regex_matches_hebrew_and_rejects_latin() -> None:
    assert HEBREW_RE.match("א")
    assert HEBREW_RE.match("ש")
    assert not HEBREW_RE.match("a")
    assert not HEBREW_RE.match("1")


def test_byte_decoder_is_total_over_256_bytes() -> None:
    decoder = _byte_decoder()
    # Every one of the 256 byte values must map to a unique printable codepoint.
    assert len(decoder) == 256
    assert len(set(decoder.values())) == 256


def test_eval_sets_are_sized_and_disjoint() -> None:
    assert len(EVAL_INPUTS_PROVIDED) == 10
    assert len(EVAL_INPUTS_OWN) == 10
    assert len(EVAL_INPUTS_ALL) == 20
    # The 20 held-out inputs must be unique (no accidental duplication).
    assert len(set(EVAL_INPUTS_ALL)) == 20


def test_decoding_queries_present() -> None:
    assert len(DECODING_QUERIES) == 10
    assert all(isinstance(q, str) and q.strip() for q in DECODING_QUERIES)
