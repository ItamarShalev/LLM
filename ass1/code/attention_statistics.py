from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch



dirpath = Path(__file__).parent.parent

graphs_path = dirpath / "graphs"
graphs_path.mkdir(exist_ok=True, parents=True)

heat_map_path = graphs_path / "heat_maps"
heat_map_path.mkdir(exist_ok=True, parents=True)
previous_token_head_checker_path = graphs_path / "previous_token_head_checker"
previous_token_head_checker_path.mkdir(exist_ok=True, parents=True)
induction_heads_checker_path = graphs_path / "induction_heads_checker"
induction_heads_checker_path.mkdir(exist_ok=True, parents=True)
vowel_consonant_head_checker_path = graphs_path / "vowel_consonant_head_checker"
vowel_consonant_head_checker_path.mkdir(exist_ok=True, parents=True)


def produce_heat_map(attention_heads, letters: str, layer_name: str = ""):
    """produce a heat map for a specific layer across all attention heads each on their own subplot

    Args:
        attention_heads (_type_): a tensor of shape (n_heads, seq_len, seq_len) containing the attention scores for a specific layer across all attention heads
        letters (str): the letters corresponding to the sequence length dimension of the attention heads, used for labeling the axes of the heat map
        layer_name (str, optional): the name of the layer for which the heat map is being produced, used for the title of the heat map. Defaults to "".
    """

    n_heads, seq_len, _ = attention_heads.shape

    # Keep up to 4 heatmaps per row, but create only the needed axes per row.
    max_cols = min(4, n_heads)
    n_rows = (n_heads + max_cols - 1) // max_cols
    fig = plt.figure(figsize=(5 * max_cols, 5 * n_rows))
    outer_grid = fig.add_gridspec(n_rows, 1)

    axes = []
    plotted_heads = 0
    for row in range(n_rows):
        heads_left = n_heads - plotted_heads
        n_cols_in_row = min(max_cols, heads_left)
        row_grid = outer_grid[row].subgridspec(1, n_cols_in_row)
        for col in range(n_cols_in_row):
            axes.append(fig.add_subplot(row_grid[0, col]))
            plotted_heads += 1
    fig.suptitle(f"Attention heads for layer {layer_name}, with '{letters}' tokens", fontsize=16)
    for i in range(n_heads):
        ax = axes[i]
        im = ax.imshow(attention_heads[i].cpu().numpy(), cmap='viridis')
        ax.set_xticks(np.arange(seq_len))
        ax.set_yticks(np.arange(seq_len))
        ax.set_xticklabels(list(letters))
        ax.set_yticklabels(list(letters))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title(f"Head {i}")
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    # save the heat map to a file
    heat_map_file = heat_map_path / f"heat_map_{layer_name}_{letters}.png"
    fig.savefig(heat_map_file)


def previous_token_head_checker(attention_head: torch.Tensor, layer: str = ""):
    """creates a plot of the average score per head of previous token score, this will tell us if a head acts as a copy mechanism

    Args:
        attention_head (torch.tensor): a tensor of shape (B, H, N, N) containing the attention scores for a specific layer across all attention heads, where B is the batch size, H is the number of heads, and N is the sequence length
        layer (str, optional): the name of the layer for which the plot is being produced, used for the title of the plot. Defaults to "".
    """

    # Use only the subdiagonal entries (i, i-1), then average across positions and batch.
    if attention_head.size(-1) < 2:
        raise ValueError("Sequence length must be at least 2 to compute previous-token attention.")

    prev_token_scores = (
        attention_head[:, :, 1:, :-1]
        .diagonal(dim1=-2, dim2=-1)
        .mean(dim=-1)
        .mean(dim=0)
    )  # shape: (H,)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(prev_token_scores)), prev_token_scores.cpu().numpy())
    plt.xlabel("Head")
    plt.ylim(0, 1)
    plt.ylabel("Average previous token score")
    plt.title(f"Average previous token score per head for layer {layer}")
    plt.xticks(range(len(prev_token_scores)))
    path = previous_token_head_checker_path / f"previous_token_head_checker_{layer}.png"
    plt.savefig(path)
    plt.show()

def begin_of_sequence_head_checker(attention_head: torch.Tensor, layer: str = ""):
    """returns the average score of the first token of a sequence across the attention scores of all tokens
       per layer per head, this will tell us if a head attends to the beginning of sequence token
       we look at the last few tokens, since they have the full sequence available to attend to the beginning of sequence token, and we average across them and across the batch dimension

    Args:
        attention_head (torch.Tensor): a tensor of shape (B, H, N, N) containing the attention scores for a specific layer across all attention heads, where B is the batch size, H is the number of heads, and N is the sequence length
        layer (str, optional): the name of the layer for which the plot is being produced, used for the title of the plot. Defaults to "".

    """

    bos_token_scores = attention_head[:, :, -3:, 0].mean(dim=-1).mean(dim=0)  # shape: (H,) #we take the last 3 tokens, as they have more context to attend
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(bos_token_scores)), bos_token_scores.cpu().numpy())
    plt.xlabel("Head")
    plt.ylim(0, 1)
    plt.ylabel("Average BOS token score")
    plt.title(f"Average BOS token score per head for layer {layer}")
    plt.xticks(range(len(bos_token_scores)))
    path = previous_token_head_checker_path / f"begin_of_sequence_head_checker_{layer}.png"
    plt.savefig(path)
    plt.show()


def induction_heads_checker(attention_head: torch.Tensor, layer: str = "", sentences: list[str] | None = None):
    """returns the average score that a token give to the same token that occured before it in a sequence
    ie AB  AB we want to see if the second B attends to the first B and by how much

    Args:
        attention_head (torch.Tensor): tensor of shape (B, H, N, N) containing the attention scores for a specific layer across all attention heads, where B is the batch size, H is the number of heads, and N is the sequence length
        layer (str, optional): the name of the layer for which the plot is being produced, used for the title of the plot. Defaults to "".
        sentences (list[str], optional): a list of sentences for which to analyze induction heads, will help determine which letters to look on which letters. Defaults to [].
    """

    batch_size, _, seq_len, seq_len_k = attention_head.shape

    repeated_token_scores = []
    for b, sentence in enumerate(sentences):
        tokens = list(sentence)

        last_position_by_token: dict[str, int] = {}
        for i, token in enumerate(tokens):
            if token in last_position_by_token:
                prev_i = last_position_by_token[token]
                repeated_token_scores.append(attention_head[b, :, i, prev_i])
            last_position_by_token[token] = i

    induction_scores = torch.stack(repeated_token_scores, dim=0).mean(dim=0)  # shape: (H,)

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(induction_scores)), induction_scores.detach().cpu().numpy())
    plt.xlabel("Head")
    plt.ylim(0, 1)
    plt.ylabel("Average induction score")
    plt.title(f"Average induction score per head for layer {layer}")
    plt.xticks(range(len(induction_scores)))
    path = induction_heads_checker_path / f"induction_heads_checker_{layer}.png"
    plt.savefig(path)
    plt.show()

def vowel_consonant_head_checker(attention_head: torch.Tensor, layer: str, sentences: list[str]):
    """analyzes the attention patterns of heads based on whether the current token is a vowel or a consonant, 
    and whether the attended tokens are vowels or consonants. It produces two subplots: one for when the current token is a vowel, 
    showing the average attention scores to vowels and consonants; and one for when the current token is a consonant, 
    showing the average attention scores to vowels and consonants. This can help identify heads that have a preference for attending to vowels or consonants based on the type of the current token.

    Args:
        attention_head (torch.Tensor): a tensor of shape (B, H, N, N) containing the attention scores for a specific layer across all attention heads, where B is the batch size, H is the number of heads, and N is the sequence length
        layer (str): the name of the layer for which the plot is being produced, used for the title of the plot.
        sentences (list[str]): a list of sentences to analyze, used to determine which tokens are vowels and consonants and to compute the average attention scores based on these categories.
    """
    vowels_set = set("aeiouAEIOU")
    consonants_set = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")
    
    batch_size, n_heads, seq_len, _ = attention_head.shape
    
    # 4 Accumulators: [Source]_[Target]_scores
    v_v_scores = torch.zeros(n_heads, device=attention_head.device)
    v_c_scores = torch.zeros(n_heads, device=attention_head.device)
    c_v_scores = torch.zeros(n_heads, device=attention_head.device)
    c_c_scores = torch.zeros(n_heads, device=attention_head.device)
    
    # Counter for denominators
    counts = {"vv": 0, "vc": 0, "cv": 0, "cc": 0}

    for b, sentence in enumerate(sentences):
        tokens = list(sentence)
        is_v = torch.tensor([t in vowels_set for t in tokens], device=attention_head.device)
        is_c = torch.tensor([t in consonants_set for t in tokens], device=attention_head.device)
        
        for i in range(1, len(tokens)):
            v_indices = torch.where(is_v[:i])[0]
            c_indices = torch.where(is_c[:i])[0]
            
            # Case 1: Current token is a Vowel
            if is_v[i]:
                if len(v_indices) > 0:
                    v_v_scores += attention_head[b, :, i, v_indices].sum(dim=-1)
                    counts["vv"] += 1
                if len(c_indices) > 0:
                    v_c_scores += attention_head[b, :, i, c_indices].sum(dim=-1)
                    counts["vc"] += 1
            
            # Case 2: Current token is a Consonant
            elif is_c[i]:
                if len(v_indices) > 0:
                    c_v_scores += attention_head[b, :, i, v_indices].sum(dim=-1)
                    counts["cv"] += 1
                if len(c_indices) > 0:
                    c_c_scores += attention_head[b, :, i, c_indices].sum(dim=-1)
                    counts["cc"] += 1

    # Calculate averages
    avg_vv = (v_v_scores / counts["vv"]).cpu().numpy() if counts["vv"] > 0 else np.zeros(n_heads)
    avg_vc = (v_c_scores / counts["vc"]).cpu().numpy() if counts["vc"] > 0 else np.zeros(n_heads)
    avg_cv = (c_v_scores / counts["cv"]).cpu().numpy() if counts["cv"] > 0 else np.zeros(n_heads)
    avg_cc = (c_c_scores / counts["cc"]).cpu().numpy() if counts["cc"] > 0 else np.zeros(n_heads)

    # Plotting with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    x = np.arange(n_heads)
    width = 0.35

    # Subplot 1: When the current token is a Vowel
    ax1.bar(x - width/2, avg_vv, width, label='Attending to Vowels', color='#4A90E2')
    ax1.bar(x + width/2, avg_vc, width, label='Attending to Consonants', color='#50E3C2')
    ax1.set_title(f"Source: Vowels (Layer {layer})")
    ax1.set_ylabel("Mean Attention")
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # Subplot 2: When the current token is a Consonant
    ax2.bar(x - width/2, avg_cv, width, label='Attending to Vowels', color='#D0021B')
    ax2.bar(x + width/2, avg_cc, width, label='Attending to Consonants', color='#F5A623')
    ax2.set_title(f"Source: Consonants (Layer {layer})")
    ax2.set_ylabel("Mean Attention")
    ax2.set_xlabel("Head Index")
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    plt.xticks(x)
    plt.tight_layout()
    plt.savefig(vowel_consonant_head_checker_path / f"vowel_consonant_head_checker_{layer}.png")
    plt.show()



__all__ = [
    "produce_heat_map",
    "previous_token_head_checker",
    "begin_of_sequence_head_checker",
    "induction_heads_checker",
    "vowel_consonant_head_checker"
]