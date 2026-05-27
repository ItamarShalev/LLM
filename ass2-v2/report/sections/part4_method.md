### Goal
Fine-tune Qwen2.5-1.5B-Instruct so that it answers English instructions in Hebrew, with a relevant, genuine answer rather than a fixed canned string. The only desired change is the output language; the content must stay correct and on topic.

### Data creation
We build a chat-format training set (messages) where each example is an English instruction as input and a high-quality Hebrew answer as output. The data is produced through two paths. An offline path uses a seed of 16 hand-written examples for development and testing without network access. An online path uses GPT (default model gpt-5.4-mini, overridable via an environment variable) to generate a large set of instruction-answer pairs across diverse categories (explanations, practical how-to, creative writing, summaries, polite refusals, and more).

Two filters run over all data. A strict leakage filter removes any example whose input appears in any of the 20 evaluation inputs, so no test question leaks into training. A language filter removes answers that are not primarily Hebrew (fewer than 40 percent Hebrew letters). This is the distinction between the naive approach, which ignores constraints, and the optimal approach, which filters.

### Training
Lightweight LoRA training with peft on Qwen2.5-1.5B-Instruct. We use the model's own chat template and mask the prompt tokens to -100 so the loss is computed only on the Hebrew assistant answer (standard SFT). Training runs on a single modest GPU (a Colab T4 is enough for a 1.5B model with LoRA). The adapter is saved to outputs/lora_adapter.

### Evaluation
evaluate.py runs the base model and the fine-tuned model on all 20 evaluation inputs (10 provided in the assignment and 10 of our own), none of which appear in training. For each input we store the base output, the fine-tuned output, and an automatic note that includes the fraction of Hebrew letters in each answer plus a short verdict. This demonstrates directly that the fine-tune switched the output language to Hebrew while preserving relevance to the content, and that the answers are varied and on-topic rather than a fixed sentence.
