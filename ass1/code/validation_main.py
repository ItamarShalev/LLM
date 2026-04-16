from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Literal

import torch
from torch import optim

import data
import lm
from transformer import TransformerLM
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _list_corpus_files(data_path: Path) -> list[Path]:
    return [file_path for file_path in sorted(data_path.glob("*.txt")) if not file_path.name.startswith("._")]


def _extract_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    current_paragraph_lines: list[str] = []

    for line in text.splitlines():
        if line.strip() == "":
            if current_paragraph_lines:
                paragraphs.append("\n".join(current_paragraph_lines))
                current_paragraph_lines = []
        else:
            current_paragraph_lines.append(line)

    if current_paragraph_lines:
        paragraphs.append("\n".join(current_paragraph_lines))

    return paragraphs


def _read_corpus_paragraphs(data_path: Path) -> list[str]:
    paragraphs: list[str] = []
    for file_path in _list_corpus_files(data_path):
        text = file_path.read_text(encoding="utf-8")
        if text:
            paragraphs.extend(_extract_paragraphs(text))

    if not paragraphs:
        raise FileNotFoundError(f"No corpus files were found in {data_path}")

    return paragraphs


def _split_into_windows(paragraphs: list[str], window_size: int, validation_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    if len(paragraphs) < 2:
        raise ValueError("Need at least 2 paragraphs to build a 90/10 train/validation split")

    rng = random.Random(seed)
    rng.shuffle(paragraphs)

    validation_size = max(1, int(len(paragraphs) * validation_fraction))
    validation_size = min(validation_size, len(paragraphs) - 1)

    validation_paragraphs = paragraphs[:validation_size]
    train_paragraphs = paragraphs[validation_size:]

    # Keep compatibility with the rest of the file, which expects list[str].
    return train_paragraphs, validation_paragraphs


def _tokenize_windows(tokenizer: data.CharTokenizer, windows: list[str]) -> list[list[int]]:
    return [tokenizer.tokenize(window) for window in windows]


def _random_order_data_iterator(sequences: list[list[int]], desired_length: int, seed: int):
    eligible_sequences = [seq for seq in sequences if len(seq) >= desired_length]
    if not eligible_sequences:
        raise ValueError(
            f"No training sequences are long enough for desired_length={desired_length}. "
            "Try reducing seq_len or changing split windowing."
        )

    rng = random.Random(seed)
    while True:
        seq = rng.choice(eligible_sequences)
        if len(seq) == desired_length:
            start_idx = 0
        else:
            start_idx = rng.randint(0, len(seq) - desired_length)
        yield seq[start_idx : start_idx + desired_length]


def _iter_fixed_length_chunks(sequences: list[list[int]], desired_length: int):
    for seq in sequences:
        if len(seq) < desired_length:
            continue
        for start_idx in range(0, len(seq) - desired_length + 1, desired_length):
            yield seq[start_idx : start_idx + desired_length]


def _evaluate(model: torch.nn.Module, validation_sequences: list[list[int]], batch_size: int, desired_length: int) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    validation_iter = _iter_fixed_length_chunks(validation_sequences, desired_length)

    with torch.no_grad():
        for batch in data.batch_items(validation_iter, batch_size):
            batch_x, batch_y = lm.batch_to_labeled_samples(batch.to(DEVICE))
            batch_x = batch_x.contiguous()
            batch_y = batch_y.contiguous()
            logits = model(batch_x)
            loss = lm.compute_loss(logits, batch_y)
            total_loss += loss.item()
            total_batches += 1

    model.train()

    if total_batches == 0:
        return float("inf")

    return total_loss / total_batches


def _sample_hyperparameters(rng: random.Random) -> dict[str, float | int]:
    seq_len = rng.choice([32, 64, 128, 256])
    n_layers = rng.choice([6, 7, 8])
    n_heads = rng.choice([6, 7, 8])
    embed_size = 32 * n_heads
    mlp_hidden_size = embed_size * 4
    learning_rate = rng.uniform(1e-4, 5e-4)

    return {
        "seq_len": seq_len,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "embed_size": embed_size,
        "mlp_hidden_size": mlp_hidden_size,
        "learning_rate": learning_rate,
    }


def _run_single_trial(
    *,
    lang: Literal["en", "he"],
    trial_index: int,
    num_batches_to_train: int,
    batch_size: int,
    efficient: bool,
    seed: int,
    gradient_clipping: float,
    save_checkpoint_every: int,
    validation_lang_path: Path,
    loss_data_path: Path,
    tokenizer: data.CharTokenizer,
    tokenized_train_data: list[list[int]],
    tokenized_validation_data: list[list[int]],
    word: str,
    hyperparams: dict[str, float | int],
) -> dict:
    seq_len = int(hyperparams["seq_len"])
    n_layers = int(hyperparams["n_layers"])
    n_heads = int(hyperparams["n_heads"])
    embed_size = int(hyperparams["embed_size"])
    mlp_hidden_size = int(hyperparams["mlp_hidden_size"])
    learning_rate = float(hyperparams["learning_rate"])

    print(
        f"[Trial {trial_index}] Starting: seq_len={seq_len}, n_layers={n_layers}, "
        f"n_heads={n_heads}, embed_size={embed_size}, mlp_hidden_size={mlp_hidden_size}, "
        f"learning_rate={learning_rate:.8f}"
    )

    train_data_iter = _random_order_data_iterator(tokenized_train_data, seq_len + 1, seed + trial_index)

    model: torch.nn.Module = TransformerLM(
        n_layers,
        n_heads,
        embed_size,
        seq_len,
        tokenizer.vocab_size(),
        mlp_hidden_size,
        with_residuals=True,
        efficient=efficient,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.9, 0.95])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_batches_to_train)

    train_losses = []
    val_losses = []
    model.train()
    best_validation_loss = float("inf")
    best_state = None
    overfit_model = None
    num_batches = 0
    early_exit = False
    start_time = time.time()

    while True:
        for batch in data.batch_items(train_data_iter, batch_size):
            batch_x, batch_y = lm.batch_to_labeled_samples(batch.to(DEVICE))
            batch_x = batch_x.contiguous()
            batch_y = batch_y.contiguous()

            logits = model(batch_x)
            loss = lm.compute_loss(logits, batch_y)

            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            num_batches += 1

            if num_batches % 10 == 0:
                validation_loss = _evaluate(model, tokenized_validation_data, batch_size, seq_len + 1)
                print(
                    f"[Trial {trial_index}] Seen {num_batches} batches. "
                    f"train_loss={loss.item():.4f}, validation_loss={validation_loss:.4f}"
                )

                train_losses.append(loss.item())
                val_losses.append(validation_loss)

                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_state = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "num_batches": num_batches,
                        "train_loss": loss.item(),
                        "validation_loss": validation_loss,
                        "best_validation_loss": best_validation_loss,
                        "hyperparams": {
                            "seq_len": seq_len,
                            "n_layers": n_layers,
                            "n_heads": n_heads,
                            "embed_size": embed_size,
                            "mlp_hidden_size": mlp_hidden_size,
                            "learning_rate": learning_rate,
                        },
                        "trial_index": trial_index,
                    }
                elif validation_loss > best_validation_loss * 1.5:
                    overfit_model = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "num_batches": num_batches,
                        "train_loss": loss.item(),
                        "validation_loss": validation_loss,
                        "hyperparams": {
                            "seq_len": seq_len,
                            "n_layers": n_layers,
                            "n_heads": n_heads,
                            "embed_size": embed_size,
                            "mlp_hidden_size": mlp_hidden_size,
                            "learning_rate": learning_rate,
                        },
                        "trial_index": trial_index,
                    }
                    print(f"[Trial {trial_index}] Validation loss increased significantly. Early stopping.")
                    early_exit = True

            if num_batches % 100 == 0:
                model.eval()
                sampled = tokenizer.detokenize(
                    model.better_sample_continuation(tokenizer.tokenize(word), max_tokens_to_generate=500, temperature=0.5, topK=5)
                )
                model.train()
                print(f"[Trial {trial_index}] Model sample: '''{sampled}'''")
                print("")

            if num_batches % save_checkpoint_every == 0:
                checkpoint_file = validation_lang_path / f"trial_{trial_index}_checkpoint_{efficient=}_{num_batches}.pt"
                state = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "num_batches": num_batches,
                    "loss": loss.item(),
                    "best_validation_loss": best_validation_loss,
                    "hyperparams": {
                        "seq_len": seq_len,
                        "n_layers": n_layers,
                        "n_heads": n_heads,
                        "embed_size": embed_size,
                        "mlp_hidden_size": mlp_hidden_size,
                        "learning_rate": learning_rate,
                    },
                    "trial_index": trial_index,
                }
                torch.save(state, checkpoint_file)
                print(f"[Trial {trial_index}] Saved checkpoint to {checkpoint_file}")

            if num_batches >= num_batches_to_train or early_exit:
                end = time.time()
                plt.figure(figsize=(10, 5))
                plt.plot(train_losses, label="Training Loss")
                plt.plot(val_losses, label="Validation Loss")
                plt.xlabel("Evaluation Steps")
                plt.ylabel("Loss")
                plt.title(
                    f"Trial {trial_index} Loss (efficient={efficient}, {lang}, seq_len={seq_len}, "
                    f"n_layers={n_layers}, n_heads={n_heads}, embed_size={embed_size}, "
                    f"learning_rate={learning_rate:.6f}, cosine annealing)",
                    wrap=True,
                )
                plt.legend()
                plt.tight_layout()
                plt.savefig(loss_data_path / f"loss_plot_trial_{trial_index}_{efficient=}_{lang}.png", bbox_inches="tight")
                plt.close()

                print(f"[Trial {trial_index}] Finished training {num_batches} batches in {end - start_time:.2f} seconds.")
                print(f"[Trial {trial_index}] Best validation loss: {best_validation_loss:.4f}")

                if early_exit and overfit_model is not None:
                    overfit_model_file = validation_lang_path / f"trial_{trial_index}_overfit_{lang}_{efficient=}.pt"
                    torch.save(overfit_model, overfit_model_file)
                    print(f"[Trial {trial_index}] Saved overfitting checkpoint to {overfit_model_file}")

                if best_state is None:
                    raise RuntimeError(f"[Trial {trial_index}] No validation result was produced.")

                return {
                    "trial_index": trial_index,
                    "best_validation_loss": best_validation_loss,
                    "best_state": best_state,
                    "hyperparams": {
                        "seq_len": seq_len,
                        "n_layers": n_layers,
                        "n_heads": n_heads,
                        "embed_size": embed_size,
                        "mlp_hidden_size": mlp_hidden_size,
                        "learning_rate": learning_rate,
                    },
                }


def main(lang: Literal["en", "he"] = "en"):


    batch_size = 64
    validation_fraction = 0.1
    num_batches_to_train = 50000
    num_trials = 10

    dirpath = Path(__file__).parent.parent
    validation_checkpoint_dir = dirpath / "validation_checkpoints_random_search"
    data_path = dirpath / "data" / lang
    loss_data_path = dirpath / "loss_tracking"
    loss_data_path.mkdir(exist_ok=True, parents=True)

    validation_checkpoint_dir.mkdir(exist_ok=True, parents=True)
    validation_lang_path = validation_checkpoint_dir / lang
    validation_lang_path.mkdir(exist_ok=True, parents=True)

    efficient = True
    save_checkpoint_every = 1000
    seed = 0

    word = "hello" if lang == "en" else "שלום"

    gradient_clipping = 1.0

    max_seq_len = 256

    corpus_files = _list_corpus_files(data_path)
    corpus_paragraphs = _read_corpus_paragraphs(data_path)
    corpus_text = "\n\n".join(corpus_paragraphs)
    train_windows, validation_windows = _split_into_windows(corpus_paragraphs, max_seq_len + 1, validation_fraction, seed)

    print(
        f"Loaded language={lang}: files={len(corpus_files)}, "
        f"paragraphs={len(corpus_paragraphs)}, "
        f"train_paragraphs={len(train_windows)}, validation_paragraphs={len(validation_windows)}"
    )

    tokenizer = data.CharTokenizer()
    tokenizer.train([corpus_text])

    tokenized_train_data = _tokenize_windows(tokenizer, train_windows)
    tokenized_validation_data = _tokenize_windows(tokenizer, validation_windows)

    search_rng = random.Random(seed)
    global_best_validation_loss = float("inf")
    global_best_state = None
    global_best_hyperparams = None

    for trial_index in range(1, num_trials + 1):
        trial_hyperparams = _sample_hyperparameters(search_rng)
        trial_result = _run_single_trial(
            lang=lang,
            trial_index=trial_index,
            num_batches_to_train=num_batches_to_train,
            batch_size=batch_size,
            efficient=efficient,
            seed=seed,
            gradient_clipping=gradient_clipping,
            save_checkpoint_every=save_checkpoint_every,
            validation_lang_path=validation_lang_path,
            loss_data_path=loss_data_path,
            tokenizer=tokenizer,
            tokenized_train_data=tokenized_train_data,
            tokenized_validation_data=tokenized_validation_data,
            word=word,
            hyperparams=trial_hyperparams,
        )

        trial_best_validation = float(trial_result["best_validation_loss"])
        if trial_best_validation < global_best_validation_loss:
            global_best_validation_loss = trial_best_validation
            global_best_state = trial_result["best_state"]
            global_best_hyperparams = trial_result["hyperparams"]

    if global_best_state is None or global_best_hyperparams is None:
        raise RuntimeError("Random search did not produce any valid trial result.")

    print(f"Random search complete. Best validation loss: {global_best_validation_loss:.4f}")
    print(
        "Best hyperparameters: "
        f"seq_len={global_best_hyperparams['seq_len']}, "
        f"n_layers={global_best_hyperparams['n_layers']}, "
        f"n_heads={global_best_hyperparams['n_heads']}, "
        f"embed_size={global_best_hyperparams['embed_size']}, "
        f"mlp_hidden_size={global_best_hyperparams['mlp_hidden_size']}, "
        f"learning_rate={global_best_hyperparams['learning_rate']:.8f}"
    )

    best_checkpoint_file = (
        validation_lang_path
        / (
            f"best_{lang}_{efficient=}"
            f"_seq{global_best_hyperparams['seq_len']}"
            f"_layers{global_best_hyperparams['n_layers']}"
            f"_heads{global_best_hyperparams['n_heads']}"
            f"_embed{global_best_hyperparams['embed_size']}"
            f"_mlp{global_best_hyperparams['mlp_hidden_size']}"
            f"_lr{global_best_hyperparams['learning_rate']:.8f}.pt"
        )
    )
    torch.save(global_best_state, best_checkpoint_file)
    print(f"Saved global best checkpoint to {best_checkpoint_file}")
    return


if __name__ == "__main__":
    #main("en")
    main("he")
