import torch
from pathlib import Path
from transformer import TransformerLM
import data
from hooks import attention_hook, ATTENTION_HEADS
from attention_statistics import induction_heads_checker, produce_heat_map, previous_token_head_checker, begin_of_sequence_head_checker

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dirpath = Path(__file__).parent.parent
checkpoint_path = dirpath / "new_checkpoints" / "en" / "checkpoint_efficient=True_50000.pt"

def main():

    seq_len = 128
    efficient = True
    n_layers = 7
    n_heads = 8
    embed_size = 256
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

    words = ["quick", "brown", "jumps", "hello", "world", "apple", "grape", "peach", "mango", "berry"]
    tokenized_words = [torch.tensor(tokenizer.tokenize(word)).to(DEVICE) for word in words]  

    for word, tokenized_word in zip(words, tokenized_words):
        model(tokenized_word.unsqueeze(0))  # Add batch dimension
        for layer_name, attention_heads in ATTENTION_HEADS.items():
            produce_heat_map(attention_heads.squeeze(), word, layer_name=layer_name)

    model(torch.stack(tokenized_words))
    for layer_name, attention_heads in ATTENTION_HEADS.items():
         previous_token_head_checker(attention_heads, layer=layer_name)

    for layer_name, attention_heads in ATTENTION_HEADS.items():
        begin_of_sequence_head_checker(attention_heads, layer=layer_name)

    sentences_with_repeated_tokens = [
    "apple grape", 
    "grape apple",   
    "hello hello", 
    "world world",  
    "model model", 
    "train train",
    "datum datum",
    "seven seven",
    "eight eight",

    ]
    tokenized_sentences = [torch.tensor(tokenizer.tokenize(sentence)).to(DEVICE) for sentence in sentences_with_repeated_tokens]
    model(torch.stack(tokenized_sentences))
    for layer_name, attention_heads in ATTENTION_HEADS.items():
        induction_heads_checker(attention_heads, layer=layer_name, sentences=sentences_with_repeated_tokens)


if __name__ == "__main__":
    main()
