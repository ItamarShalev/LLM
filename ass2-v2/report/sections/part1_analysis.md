### Cross-model analysis and trends

**Universal choices (full consensus across all ten models).**

- Decoder-only Transformer with RoPE positional encoding. No model in the set uses learned or sinusoidal absolute positions.
- RMSNorm rather than LayerNorm.
- A gated SwiGLU MLP built on the SiLU activation. The activation is the same in every block of every model; nobody mixes activation types.
- Untied or tied embeddings vary, but the embedding and unembedding share the same hidden width.

**Choices with no consensus.**

- Attention scheme. Grouped-query attention (GQA, num_kv_heads < num_heads) is now the majority: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, granite-3.3-8b-instruct, Phi-4-mini-instruct, Falcon3-7B-Instruct, dictalm2.0-instruct. Full multi-head attention survives only in OLMo-2-1124-7B-Instruct, SmolLM2-1.7B-Instruct. DeepSeek-V3 replaces both with Multi-head Latent Attention (MLA).
- Vocabulary size spans a 6x range, from 49,152 (SmolLM2-1.7B-Instruct) to 131,072 (Falcon3-7B-Instruct).
- Context length spans from 4,096 (OLMo-2-1124-7B-Instruct) to 163,840 (DeepSeek-V3), often through RoPE scaling (YaRN, llama3, longrope) rather than native training length.
- RoPE theta ranges over four orders of magnitude (10,000 to 10,000,000), tracking the target context length.

**Deviating models.**

- OLMo-2 is the clearest outlier on normalization: it uses a reordered post-norm scheme (norm after the attention and MLP sublayers) plus QK-norm, where every other model is standard pre-norm.
- DeepSeek-V3 is the structural outlier: a 256-expert + 1-shared-expert MoE with MLA attention and the only model here that is not a dense decoder.
- Granite multiplies embeddings, attention scores, residuals and logits by fixed scalars (embedding_multiplier, attention_multiplier, residual_multiplier, logits_scaling); the others do not.
- Phi-4-mini and DeepSeek-V3 use partial / decoupled RoPE (only part of each head is rotated); the others rotate the whole head.

**Sizing rules of thumb that hold.**

- MLP-to-hidden ratio clusters tightly: mean 3.97x, range 2.67x to 7.5x. The SwiGLU models that target ~2.67x (8/3) are visible, while a few widen the MLP toward 3.5x to 5x to add capacity without more layers.
- head_dim is almost an invariant: the observed values are ['128', '256', '64', 'MLA qk=192, v=128']. 64 or 128 dominate.
- hidden_size is always a multiple of num_attention_heads (so head_dim is integral) and is a power-of-two-friendly number.
- Larger vocabularies pair with the byte-level BPE tokenizers; the SentencePiece models (Mistral, DictaLM) keep ~32k.

**If I were to build a new ~7-8B model.**

Following the consensus and the rules of thumb above: decoder-only, pre-norm RMSNorm, RoPE with a theta chosen for the target context, SwiGLU/SiLU MLP at roughly 3.5x the hidden size, head_dim of 128, and GQA with 8 KV heads (a strong quality-to-KV-cache trade-off shared by Llama, Mistral, Granite and DictaLM). I would pick hidden_size 4096, 32 layers, 32 query heads and 8 KV heads, a 128k byte-level BPE vocabulary if multilingual coverage matters, and reserve MoE/MLA for a later scale-up where the inference-time savings justify the extra system complexity.
