"""
Shared token/text utilities used by Part 2 (tokenizer analysis) and Part 3
(Hebrew-allowed token identification + constrained decoding).

The tricky part here is figuring out what a single token actually “means” in raw
bytes or text, since different tokenizer families handle this very differently:

* Byte-level BPE (GPT-2 style: Llama, Qwen, OLMo-2, Granite, Falcon3, SmolLM2,
  Phi-4, DeepSeek). In this setup, each byte (0–255) is mapped to a printable
  Unicode character using GPT-2’s `bytes_to_unicode` scheme. A leading space is
  typically represented as 'Ġ' (U+0120). To reconstruct the original bytes, we
  invert that mapping. Because Hebrew text is UTF-8 encoded, a single token may
  correspond to a full Hebrew word, a subword fragment, or just a single byte.

* SentencePiece with byte fallback (Mistral v0.3, DictaLM). Normal tokens use the
  '▁' (U+2581) character to indicate a leading space. For rare or unknown pieces,
  the tokenizer falls back to explicit byte tokens like '<0xD7>'. Hebrew text is
  often assembled from these byte-level fallback tokens.

`token_to_bytes` returns the exact byte sequence produced by a token id (or None
for structural/special tokens that don’t map to text). `is_hebrew_participating`
is used to decide which tokens can participate in Hebrew generation.
"""

from __future__ import annotations

import functools
import re
import unicodedata

# Hebrew Unicode block (letters, final forms, points, punctuation like geresh).
HEBREW_RE = re.compile(r"[\u0590-\u05FF\uFB1D-\uFB4F]")

# Characters we explicitly allow inside an otherwise-Hebrew token: ASCII digits,
# common punctuation, whitespace, and the two word-boundary meta symbols.
# En dash and em dash are intentional here: they are legitimate punctuation in
# Hebrew text that we want to permit inside an otherwise-Hebrew token.
ALLOWED_NONLETTER = set(" \t\n0123456789.,!?;:'\"()[]-–—…/%₪\u2581\u0120\u010a")  # noqa: RUF001

# Scripts that, if present, disqualify a token from the Hebrew set.
FOREIGN_LETTER_RE = re.compile(
    r"[A-Za-z\u00C0-\u024F\u0400-\u04FF\u0600-\u06FF\u0370-\u03FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]"
)


@functools.lru_cache(maxsize=1)
def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode: printable-unicode-char -> byte."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


def detect_family(tokenizer) -> str:
    """Return 'byte_level' or 'sentencepiece'. The authoritative signal is the
    backend decoder type; sampling the vocab is only a fallback."""
    try:
        dec = type(tokenizer.backend_tokenizer.decoder).__name__.lower()
        if "bytelevel" in dec or "byte_level" in dec:
            return "byte_level"
        if "metaspace" in dec or "sentencepiece" in dec or "bpedecoder" in dec:
            return "sentencepiece"
    except Exception:
        pass
    sample = tokenizer.convert_ids_to_tokens(list(range(min(2000, tokenizer.vocab_size))))
    text = "".join(t for t in sample if t)
    if "\u0120" in text or "\u010a" in text:  # Ġ / Ċ byte-level markers
        return "byte_level"
    if "\u2581" in text:  # ▁ meta space
        return "sentencepiece"
    return "byte_level"


_HEX_FALLBACK = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def token_to_bytes(tokenizer, token_id: int, family: str) -> bytes | None:
    """Raw bytes a token id contributes, or None for special/structural tokens."""
    tok = tokenizer.convert_ids_to_tokens(token_id)
    if tok is None:
        return None
    # Special / added tokens (e.g. <|im_start|>, <s>, [INST]) carry no surface bytes.
    if token_id in getattr(tokenizer, "all_special_ids", []):
        return None
    if tok.startswith("<") and tok.endswith(">") and not _HEX_FALLBACK.match(tok):
        return None

    if family == "byte_level":
        dec = _byte_decoder()
        try:
            return bytes(dec[c] for c in tok)
        except KeyError:
            return None
    # sentencepiece
    m = _HEX_FALLBACK.match(tok)
    if m:
        return bytes([int(m.group(1), 16)])
    return tok.replace("\u2581", " ").encode("utf-8")


def token_surface(tokenizer, token_id: int, family: str) -> str | None:
    """Best-effort decoded text of a token (may be a partial UTF-8 fragment)."""
    b = token_to_bytes(tokenizer, token_id, family)
    if b is None:
        return None
    return b.decode("utf-8", errors="replace")


def is_hebrew_participating(tokenizer, token_id: int, family: str) -> bool:
    """True if the token may take part in Hebrew text.

    Rules:
      * include a token whose decoded text contains a Hebrew character and no
        letter from another script;
      * include pure punctuation / digit / whitespace tokens (they appear inside
        Hebrew sentences);
      * include byte-fragment tokens whose bytes are valid lead/continuation
        bytes of the Hebrew UTF-8 range (so Hebrew letters split across tokens can
        still be assembled), as long as they decode without foreign letters.
    Numbers and punctuation are included by design (per the assignment); words in
    other languages are excluded.
    """
    b = token_to_bytes(tokenizer, token_id, family)
    if b is None:
        return False

    text = b.decode("utf-8", errors="replace")
    if FOREIGN_LETTER_RE.search(text):
        return False

    if HEBREW_RE.search(text):
        return True

    # Pure punctuation / digits / whitespace token.
    stripped = text.replace("\ufffd", "")
    if stripped and all(
        (c in ALLOWED_NONLETTER) or unicodedata.category(c)[0] in ("P", "N", "Z") for c in stripped
    ):
        return True

    # Byte fragment: every byte is in the UTF-8 footprint of the Hebrew block.
    # Hebrew U+0590..U+05FF -> lead bytes 0xD6/0xD7, continuation 0x80..0xBF.
    return bool(b and all(byte in range(0x80, 0xC0) or byte in (0xD6, 0xD7) for byte in b))
