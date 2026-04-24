import torch
from transformer import TransformerLM
from pathlib import Path
import data
from tqdm import tqdm

en_tokenizer, _ = data.load_data(Path("ass1/data/en"))
hebrew_tokenizer, _ = data.load_data(Path("ass1/data/he"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#load the final english model made in main.py

checkpoint_path = "ass1/new_checkpoints/en/checkpoint_efficient=True_50000.pt"

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

output_dir = Path("ass1/test_outputs")
output_dir.mkdir(exist_ok=True)

english_file = output_dir / "english_samples.txt"

test_words = [
    "apple", "beach", "chair", "dream", "earth", 
    "field", "glass", "house", "image", "juice", 
    "light", "money", "night", "ocean", "paper", 
    "radio", "stone", "table", "uncle", "water", 
    "break", "clean", "drink", "fight", "guess", 
    "laugh", "paint", "raise", "sleep", "teach", 
    "think", "write", "abide", "crypt", "lyric", 
    "rhyme", "syrup", "brief", "clock", "flame", 
    "proud", "shelf", "small", "track", "train", 
    "voice", "young", "basic", "clear", "final"
]

with open(english_file, "w") as f:
    for word in tqdm(test_words, desc="Generating samples for English words"):
        sample_text = en_tokenizer.detokenize(
            model.better_sample_continuation(en_tokenizer.tokenize("hello"), max_tokens_to_generate=500, temperature=0.5, topK=5)
            )
        f.write(sample_text + "\n\n\n\n")



#get last hebrew checkpoint from main.py and generate samples for hebrew as well
checkpoint_path_hebrew = "ass1/new_checkpoints/he/checkpoint_efficient=True_50000.pt"

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
test_words_hebrew = [
    "שלום", "בית", "אור", "מים", "לחם",
    "שמש", "ילד", "עיר", "הרב", "מלך",
    "ספר", "קוד", "נפש", "רוח", "חלב",
    "דבש", "ארץ", "זמן", "לבן", "שחור",
    "גדול", "קטן", "חדש", "ישן", "טוב",
    "בוקר", "ערב", "לילה", "פרח", "עץ",
    "כסף", "זהב", "דרך", "נחל", "ים",
    "אביב", "חורף", "קיץ", "סתיו", "חום",
    "קור", "מחשב", "טלפון", "מסך", "שולחן",
    "כיסא", "דלת", "חלון", "קיר", "תקרה"
]

with open(hebrew_file, "w", encoding="utf-8") as f:
    for word in tqdm(test_words_hebrew, desc="Generating samples for Hebrew words"):
        sample_text = hebrew_tokenizer.detokenize(
            model_hebrew.better_sample_continuation(hebrew_tokenizer.tokenize("שלום"), max_tokens_to_generate=500, temperature=0.5, topK=5)
            )
        f.write(sample_text + "\n\n\n\n")




    

