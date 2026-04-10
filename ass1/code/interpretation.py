import torch
from pathlib import Path
from transformer import TransformerLM
import data
from hooks import attention_hook, ATTENTION_HEADS
from attention_statistics import induction_heads_checker, produce_heat_map, previous_token_head_checker, begin_of_sequence_head_checker

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


    five_letter_words = ["quick", "brown", "jumps", "hello", "world", "apple", "grape", "peach", "mango", "berry"]
    five_letter_tokenized_words = [tokenizer.tokenize(word) for word in five_letter_words]  

    #for word, tokenized_word in zip(words, tokenized_words):

    #    model.better_sample_continuation(tokenized_word, max_tokens_to_generate=1, temperature=0.5, topK=5)
    #    for layer_name, attention_heads in ATTENTION_HEADS.items():
    #        produce_heat_map(attention_heads.squeeze(), word, layer_name=layer_name)

    model(torch.stack([torch.tensor(word) for word in five_letter_tokenized_words]).to(DEVICE))
    # for layer_name, attention_heads in ATTENTION_HEADS.items():
    #     previous_token_head_checker(attention_heads, layer=layer_name)

    for layer_name, attention_heads in ATTENTION_HEADS.items():
        begin_of_sequence_head_checker(attention_heads, layer=layer_name)

    sentences_with_repeated_tokens = [
    "apple grape", # 11
    "grape apple", # 11
    "easy peasy ", # 11 
    "peasy easy ", # 11 
    "hello hello", # 11
    "world world", # 11
    "test test  ", # 11
    "data data  ", # 11 
    "model model", # 11
    "train train"  # 11
    ]
    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences_with_repeated_tokens]
    model(torch.stack([torch.tensor(sentence) for sentence in tokenized_sentences]).to(DEVICE))
    for layer_name, attention_heads in ATTENTION_HEADS.items():
        induction_heads_checker(attention_heads, layer=layer_name, sentences=sentences_with_repeated_tokens)


if __name__ == "__main__":
    main()
