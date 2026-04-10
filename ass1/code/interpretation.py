import torch
from pathlib import Path
from transformer import TransformerLM
import data
from hooks import attention_hook, ATTENTION_HEADS
from attention_statistics import produce_heat_map 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dirpath = Path(__file__).parent.parent
checkpoint_path = dirpath / "checkpoints" / "en" / "checkpoint_efficient=True_50000.pt"

def main():

    seq_len = 128
    batch_size = 64
    efficient = True
    n_layers = 6
    n_heads = 6
    embed_size = 192
    mlp_hidden_size = embed_size * 4

    tokenizer, _ = data.load_data(dirpath / "data" / "en")

    model: torch.nn.Module = TransformerLM(
        n_layers=n_layers,
        n_heads=n_heads,
        embed_size=embed_size,
        max_context_len=seq_len,
        vocab_size=tokenizer.vocab_size(),
        mlp_hidden_size=mlp_hidden_size,
        with_residuals=True,
        efficient=efficient,    
        register_hooks=True
    ).to(DEVICE)
    model.eval()

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {checkpoint_path}.")

    words = "quick brown fox jumps over the lazy dog".split()
    tokenized_words = [tokenizer.tokenize(word) for word in words]

    for word, tokenized_word in zip(words, tokenized_words):

        model.better_sample_continuation(tokenized_word, max_tokens_to_generate=1, temperature=0.5, topK=5)
        for layer_name, attention_heads in ATTENTION_HEADS.items():
            produce_heat_map(attention_heads.squeeze(), word, layer_name=layer_name)   



if __name__ == "__main__":
    main()
