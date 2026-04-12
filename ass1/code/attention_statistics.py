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