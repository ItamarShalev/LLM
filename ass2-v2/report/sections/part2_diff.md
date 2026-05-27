### Tokenization differences

Chosen text: `Supercalifragilisticexpialidocious`

The ten tokenizers produced 7 distinct splittings of this text. Three representative tokenizations:

**Tokenization 1** (e.g. Llama-3.1-8B-Instruct, 11 tokens): Sup | erc | al | if | rag | il | istic | exp | ial | id | ocious
  Agreeing models among the other nine: 2 (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, OLMo-2-1124-7B-Instruct).

**Tokenization 2** (e.g. Mistral-7B-Instruct-v0.3, 12 tokens): ␣Super | cal | if | rag | il | ist | ice | xp | ial | id | oc | ious
  Agreeing models among the other nine: 1 (Mistral-7B-Instruct-v0.3, dictalm2.0-instruct).

**Tokenization 3** (e.g. granite-3.3-8b-instruct, 11 tokens): Super | cal | if | rag | il | istic | exp | ial | id | oc | ious
  Agreeing models among the other nine: 0 (granite-3.3-8b-instruct).

**Likely causes.** The splits diverge mainly because (a) the byte-level BPE vocabularies were learned on different corpora, so multi-digit numbers, the rare compound word and the contraction apostrophe are merged to different depths; (b) the SentencePiece models (Mistral, DictaLM) treat the leading space and punctuation differently from the byte-level models; and (c) the very large vocabularies (Phi-4 at ~200k, Qwen at ~152k) tend to keep common chunks whole where the smaller vocabularies (SmolLM2, Granite at ~49k) break them into more pieces.

### Measurement method (avg tokens per word)

A 'word' is defined as a whitespace-delimited unit: we count words with `text.split()` over a fixed prose sample (one English sample and one Hebrew sample, identical across all ten tokenizers so the numbers are comparable). For each tokenizer we encode the same sample without special tokens (`encode(text, add_special_tokens=False)`), count the resulting tokens, and report tokens divided by words. Using one shared sample removes corpus bias from the comparison: every difference in the ratio is then attributable to the tokenizer alone. Word-boundary marking is reported per model in tokenizers.csv: byte-level BPE models mark a word start with the meta byte 'Ġ' (U+0120, the byte-level rendering of a leading space), while SentencePiece models use the meta symbol '▁' (U+2581).

The Hebrew ratio is the most revealing number. English sits near 1.1 to 1.2 tokens per word for every model. Hebrew splits far more: models with dedicated Hebrew coverage (Phi-4-mini and Qwen near 1.9, DictaLM near 2.25, DeepSeek near 2.4) stay low, whereas English-centric vocabularies fall back toward bytes and balloon to 4.5 to 5.8 tokens per Hebrew word (SmolLM2 highest, Llama and OLMo-2 around 5). This is the practical cost of tokenizer coverage: the same Hebrew sentence can be three times more expensive to process on one model than another.
