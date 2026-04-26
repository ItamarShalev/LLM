"""the purpose of this file is to analyze the attention heads of the best model we trained in the previous part, 
and to produce heat maps for the attention scores of the heads. 
We will analyze the attention heads for different words and sentences, 
and check if there are any patterns in the attention scores. 
We will also check if there are any heads that attend to specific tokens in the input, 
such as the previous token or the first token in the sequence
"""


import torch
from pathlib import Path
from transformer import TransformerLM
import data
from hooks import ATTENTION_HEADS #this is the dictionary where the attention heads are stored by the hooks
from attention_statistics import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dirpath = Path(__file__).parent.parent

#this is the checkpoint of the model we will be analyzing
checkpoint_path = dirpath / "new_checkpoints" / "en" / "checkpoint_efficient=True_50000.pt"

def main():

    #this hyperparmeteters of the best model
    seq_len = 128
    efficient = True
    n_layers = 7
    n_heads = 8
    embed_size = 256
    mlp_hidden_size = embed_size * 4

    #need this for the final layer for the vocabulary as well as for tokenize the words we will analyze
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
    print(tokenizer.vocab_size())
    model.eval()


    
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {checkpoint_path}.")

    words = ["quick", "brown", "jumps", "hello", "world", "apple", "grape", "peach", "mango", "berry"]
    tokenized_words = [torch.tensor(tokenizer.tokenize(word)).to(DEVICE) for word in words]  

    for word, tokenized_word in zip(words, tokenized_words):
        model(tokenized_word.unsqueeze(0))  # Add batch dimension
        for layer_name, attention_heads in ATTENTION_HEADS.items():

            #get the attention heads for the current layer and produce a heat map for the attention scores of the first token in the word (the one that corresponds to the first character) to all other tokens in the word. The heat map should be saved as an image file with a name that includes the word and the layer name.
            produce_heat_map(attention_heads.squeeze(), word, layer_name=layer_name)
    

    model(torch.stack(tokenized_words))

    for layer_name, attention_heads in ATTENTION_HEADS.items():
         #check if there are attention heads that attend to the previous token in the sequence, and produce a heat map for the attention scores of the first token in each word to the previous token in the word. The heat map should be saved as an image file with a name that includes the word and the layer name.
         previous_token_head_checker(attention_heads, layer=layer_name)

    

    for layer_name, attention_heads in ATTENTION_HEADS.items():
        #check if there are attention heads that attend to the first token in the sequence, and produce a heat map for the attention scores of the first token in each word to the first token in the word. The heat map should be saved as an image file with a name that includes the word and the layer name.
        begin_of_sequence_head_checker(attention_heads, layer=layer_name)



    sentences_with_repeated_tokens = [
    "hello hello", 
    "world world",  
    "model model", 
    "train train",
    "datum datum",
    "seven seven",
    "eight eight",
    "truth truth",
    "false false",
    "apple apple",
    ]
    tokenized_sentences = [torch.tensor(tokenizer.tokenize(sentence)).to(DEVICE) for sentence in sentences_with_repeated_tokens]
    model(torch.stack(tokenized_sentences))
    for layer_name, attention_heads in ATTENTION_HEADS.items():
        #check induction
        induction_heads_checker(attention_heads, layer=layer_name, sentences=sentences_with_repeated_tokens)
    
    sentences = [
    "era of coast idea brand union alert ratio ", # V:17, C:18
    "open ice area unite blend ideal easy echo ", # V:20, C:16
    "blue iris area awake solid coast epic     ", # V:17, C:18
    "ideal image above eagle solid early ago   ", # V:19, C:17
    "quiet coast agile irony under solid brand ",  # V:18, C:18
    "audio image ultra coast ebook brand item  ", # V:20, C:16
    "equal solid oiler alert coast urban brand ",  # V:18, C:18
    "unite every solid email adult coast brand ",  # V:17, C:19
    "coast aside ivory awake eagle brand icon  ", # V:19, C:16
    ]


    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    tokenized_sentences = [torch.tensor(tokens).to(DEVICE) for tokens in tokenized_sentences]
    model(torch.stack(tokenized_sentences))

    for layer_name, attention_heads in ATTENTION_HEADS.items():
        #check if there are attention heads that attend differently to vowels and consonants
        vowel_consonant_head_checker(attention_heads, layer=layer_name, sentences=sentences)


if __name__ == "__main__":
    main()
