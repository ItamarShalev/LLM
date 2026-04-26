import torch
from transformer import TransformerLM
from pathlib import Path
import data
from tqdm import tqdm

DIRPATH = Path(__file__).resolve().parent.parent
en_tokenizer, _ = data.load_data(DIRPATH / "data" / "en")
hebrew_tokenizer, _ = data.load_data(DIRPATH / "data" / "he")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#load the final english model made in main.py

checkpoint_path = DIRPATH / "new_checkpoints" / "en" / "checkpoint_efficient=True_50000.pt"

model = TransformerLM(
    n_layers=7,
    n_heads=8,
    embed_size=256,
    max_context_len=128,
    vocab_size=en_tokenizer.vocab_size(),
    mlp_hidden_size=256 * 4,
    with_residuals=True,
    efficient=True
).to(device)
model.eval()
checkpoint = torch.load(checkpoint_path, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

output_dir = DIRPATH / "test_outputs"
output_dir.mkdir(exist_ok=True)

english_file = output_dir / "english_samples.txt"

with open(english_file, "w", encoding="utf-8") as f:
    for _ in tqdm(range(50), desc="Generating samples for English words"):
        sample_text = en_tokenizer.detokenize(
            model.better_sample_continuation(en_tokenizer.tokenize("hello"), max_tokens_to_generate=500, temperature=0.5, topK=5)
            )
        f.write(f'"{sample_text}"' + "\n\n")



#get last hebrew checkpoint from main.py and generate samples for hebrew as well
checkpoint_path_hebrew = DIRPATH / "new_checkpoints" / "he" / "checkpoint_efficient=True_50000.pt"

model_hebrew = TransformerLM(
    n_layers=7,
    n_heads=8,
    embed_size=256,
    max_context_len=128,
    vocab_size=hebrew_tokenizer.vocab_size(),
    mlp_hidden_size=256 * 4,
    with_residuals=True,
    efficient=True
).to(device)

model_hebrew.eval()

checkpoint_hebrew = torch.load(checkpoint_path_hebrew, weights_only=False)
model_hebrew.load_state_dict(checkpoint_hebrew["model_state_dict"])

hebrew_file = output_dir / "hebrew_samples.txt"

with open(hebrew_file, "w", encoding="utf-8") as f:
    for word in tqdm(range(50), desc="Generating samples for Hebrew words"):
        sample_text = hebrew_tokenizer.detokenize(
            model_hebrew.better_sample_continuation(hebrew_tokenizer.tokenize("שלום"), max_tokens_to_generate=500, temperature=0.5, topK=5)
            )
        f.write(f'"{sample_text}"' + "\n\n")
