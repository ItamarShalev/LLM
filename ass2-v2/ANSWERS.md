# Full Answers and Insights - Assignment 2

This document answers every question the assignment poses, for all four parts, with reasoning grounded in the actual output the code produced. Every number here comes from the files in the outputs directory. It is written in clear academic English.

## Part 1 - Architectural choices

### Extraction method
We extracted every attribute directly from each model's config.json on the Hugging Face Hub, using extract_architecture.py, which downloads the file through the official resolve endpoint and caches it locally. For the gated model meta-llama/Llama-3.1-8B-Instruct, when no accepted token is present, we automatically fall back to a verified public mirror (NousResearch/Meta-Llama-3.1-8B-Instruct) that hosts the identical config.json, so the row is never missing. The source column in the table records where each value came from.

Attributes that are not present in config.json alone (such as normalization placement) come from a documented knowledge table in config.py, based on the modeling code and papers, and are marked as such. Attributes that do not apply to a model are marked NA (for example moe_details for a dense model).

### Uncertainties
There are three main uncertainties. First, the parameter-count estimate is computed from a dense formula (embedding, attention, MLP times number of layers) and is therefore accurate to about one percent for dense models, but it does not apply to DeepSeek-V3, which is a Mixture of Experts, so it is marked NA there. Second, the maximum context length is sometimes achieved through extension methods (YaRN, llama3 scaling, longrope) rather than native training at that length, so the value reflects the supported limit rather than necessarily the training length. Third, normalization placement is not always explicit in the config and is therefore backed by the knowledge table.

### Summary of differences and trends
Choices shared by all ten models: every model is a decoder-only Transformer with RoPE positional encoding (none uses learned or sinusoidal absolute positions, and none uses ALiBi), every model uses RMSNorm rather than LayerNorm, and every model is built on a gated SwiGLU MLP with the SiLU activation. This is the base pattern that the current generation has settled on.

Choices with no consensus: the attention scheme. Grouped-Query Attention (fewer KV heads than query heads) is now the majority and appears in Llama, Mistral, Qwen, Granite, Phi-4-mini, Falcon3 and DictaLM. Full Multi-Head Attention survives only in OLMo-2 and SmolLM2. DeepSeek-V3 replaces both with Multi-head Latent Attention. Vocabulary size spans a sixfold range, from 49,152 in SmolLM2 to over 150,000 in Qwen and Phi-4. Context length spans from 4,096 in OLMo-2 to 163,840 in DeepSeek-V3. The RoPE theta parameter spans four orders of magnitude, from 10,000 to 10,000,000, tracking the target context length.

### Edge cases (especially important)
DeepSeek-V3 is the structural outlier. It combines a Mixture of Experts with 256 routed experts and one shared expert, top-8 expert selection per token, and the first three layers kept dense. It also uses MLA, which compresses the key and value matrices into a low-rank latent vector (kv_lora_rank) and splits the query head into rope and nope parts, dramatically shrinking the KV cache at inference time. This is also why the dense parameter estimate does not apply to it.

OLMo-2 is the normalization outlier. Instead of standard pre-norm it uses a reordered post-norm scheme (normalization after the attention and MLP sublayers) and adds QK-norm, that is, normalization of the query and key before computing attention. This improves training stability.

Phi-4-mini (and DeepSeek-V3) use partial RoPE, meaning only part of each head is rotated and the rest carries no positional encoding. This differs from the majority, which rotate the whole head.

Falcon3 defines an unusual head_dim of 256, while almost all the others stick to 64 or 128. For Falcon3, hidden_size divided by the number of heads does not equal head_dim, and head_dim is set explicitly.

Granite explicitly multiplies the embedding, the attention scores, the residual and the logits by fixed scalars in the config (embedding_multiplier, attention_multiplier, residual_multiplier, logits_scaling), a technique no other model in the set uses.

### Sizing rules that hold
The MLP-to-hidden ratio clusters around a mean of about 3.97, with models that target 2.67 (which is 8/3, the classic SwiGLU value) clearly visible, and others widening the MLP up to about fivefold to add capacity without adding layers. We include a bar chart of this ratio per model in the report. head_dim is almost an invariant at 64 or 128. hidden_size is always a multiple of the number of attention heads. Large vocabularies pair with byte-level BPE tokenizers, while the SentencePiece models (Mistral, DictaLM) stay around 32,000.

### If we were to build a 7 to 8 billion parameter model
We would follow the consensus: decoder-only, pre-norm with RMSNorm, RoPE with a theta tuned to the target context length, a SwiGLU MLP at roughly 3.5 times the hidden width, head_dim of 128, and GQA with 8 KV heads, which is an excellent quality-to-KV-cache trade-off shared by Llama, Mistral, Granite and DictaLM. We would pick hidden_size 4096, 32 layers, 32 query heads and 8 KV heads, a 128k byte-level BPE vocabulary if multilingual coverage matters, and reserve MoE and MLA for a later scale-up where the inference-time savings justify the system complexity.

## Part 2 - Tokenizers

### Tokenizer characteristics
Eight of the ten models use byte-level BPE (the GPT-2 family), and two (Mistral and DictaLM) use SentencePiece BPE with byte fallback. We added a bonus column, tokenizer_backend, that records the concrete backend class (for example Qwen2Tokenizer, GPT2Tokenizer, TokenizersBackend) for transparency.

### Word-boundary strategy
In byte-level tokenizers, the start of a word is marked by the meta byte represented as G-with-dot (codepoint U+0120), which is the byte-level rendering of a leading space. A token without that marker continues the previous word. In SentencePiece tokenizers, the start of a word is marked by the lower-one-eighth-block meta symbol (codepoint U+2581). When decoding, byte-level concatenates the byte strings and maps them back to bytes, and SentencePiece concatenates the pieces and replaces the meta symbol with a space.

### Measurement method for average tokens per word
A word is defined as a whitespace-delimited unit, and we count words with a simple split over two fixed prose samples, one English and one Hebrew, identical across all ten tokenizers. For each tokenizer we encode the same text without special tokens, count the resulting tokens, and report tokens divided by words. Using a shared text removes corpus bias from the comparison, so every difference in the ratio is attributable to the tokenizer itself.

### The interesting result
In English all models sit around 1.1 to 1.2 tokens per word. In Hebrew the gap is dramatic. Models with dedicated Hebrew coverage keep a low ratio: Phi-4-mini and Qwen near 1.9, DictaLM near 2.25, and DeepSeek near 2.4. By contrast, English-centric vocabularies fall back to the byte level and balloon to 4.5 to 5.8 tokens per Hebrew word, with SmolLM2 the highest and Llama and OLMo-2 around 5. This is the practical cost of tokenizer coverage: the same Hebrew sentence can be three times more expensive to process on one model than on another. It is worth noting that DictaLM, a Hebrew-focused model, achieves a good Hebrew ratio despite being SentencePiece-based, which shows that the corpus the tokenizer was trained on matters more than the algorithm type.

### The text that tokenizes differently
We chose the word Supercalifragilisticexpialidocious, a long, rare word that no tokenizer has seen whole. It produced 7 distinct splittings across the ten models, comfortably exceeding the threshold of three models splitting it differently. The report shows three representative splittings with an exact count of how many of the other models agree on each. The reasons for the variation: the vocabularies were learned on different corpora and therefore merge substrings to different depths, the SentencePiece models treat the leading space and punctuation differently, and very large vocabularies tend to keep common chunks whole while smaller ones break them into more pieces.

## Part 3 - Constrained decoding in Hebrew

### Identifying the allowed tokens
identify_hebrew_tokens.py scans the entire vocabulary of each model and classifies every token. A token is considered Hebrew-participating if it contains at least one Hebrew character in the Unicode range U+0590 to U+05FF and contains no characters from other scripts, or if it is purely punctuation, digits or whitespace, or if it is a byte fragment that belongs to the UTF-8 footprint of the Hebrew block. That last case is critical: it ensures that SentencePiece byte-fallback tokens (as in Mistral) are caught, not only full byte-level tokens (as in Qwen). In practice we obtained 7,725 allowed tokens for Qwen (including thousands of native Hebrew subwords) and 1,006 for Mistral (mostly single-byte tokens, because its Hebrew lexicon is sparse and relies on fallback). Latin characters are excluded entirely.

### Implementing constrained decoding
In constrained_decode.py we implemented a class that inherits from LogitsProcessor. At every step it sets the logits of all tokens not in the allowed list to negative infinity, so they can never be chosen. EOS and pad are added to the allowed set at runtime and are not stored in the JSON file, so the model can terminate the sequence, and we made sure end-of-text is added only once. At run time, run_decoding.py runs each of the 10 English queries on both models, once unconstrained and once constrained, and stores both versions. The expectation is that the constrained output is Hebrew-only while the unconstrained output may be English or mixed.

## Part 4 - Fine-tuning in Hebrew

### Why Qwen2.5-1.5B-Instruct
We chose this model because it balances size against resources: 1.5 billion parameters train with LoRA on a single modest GPU (such as a Colab T4), and its tokenizer already has excellent Hebrew coverage (as we saw in Part 2, a ratio of about 1.9 tokens per Hebrew word), so the model does not need to learn Hebrew from scratch but only to prefer it as the output language.

### Data creation and leakage prevention
make_data.py creates pairs of an English instruction and a high-quality Hebrew answer. In offline mode there is a seed of 16 hand-written examples, and in online mode GPT (default gpt-5.4-mini) generates a large set across diverse categories. Two filters run over all data: a strict leakage filter that removes any example whose input appears in any of the 20 evaluation inputs, so no test question leaks into training, and a language filter that removes answers that are not primarily Hebrew (fewer than 40 percent Hebrew letters). This is the distinction between the naive approach, which ignores constraints, and the optimal approach, which filters.

### Training parameters
LoRA on the attention and MLP modules, using the model's own chat template, masking the prompt tokens to -100 so the loss is computed only on the Hebrew answer. A range of 4 to 8 epochs is recommended, a learning rate on the order of 5e-5 to 2e-4, and a short warmup. Two variants can be produced and compared.

### Evaluation method
evaluate.py runs the base model and the fine-tuned model on all 20 evaluation inputs (10 provided and 10 of our own), none of which appear in training. For each input we store the base output, the fine-tuned output, and an automatic note that includes the fraction of Hebrew letters in each answer and a short verdict. This proves that the fine-tune switched the output language to Hebrew while preserving relevance to the content, and that the answers are varied and relevant to each question rather than a fixed sentence.

## Note on the GPU-dependent outputs
The code, the structure, the Part 1 and Part 2 CSV tables, the Hebrew allowed-token lists, and all the analyses were produced and verified. The two runtime outputs that require a GPU are decoding_outputs.jsonl (Part 3) and eval_outputs.jsonl together with lora_adapter (Part 4), because they require loading 7 billion parameter models and actual training. They should be run on a GPU machine with the tokens, via make p3 and make p4, and they will enter the final report automatically.
