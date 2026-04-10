from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


dirpath = Path(__file__).parent.parent

heat_map_path = dirpath / "heat_maps"
heat_map_path.mkdir(exist_ok=True, parents=True)


def produce_heat_map(attention_heads, letters: str, layer_name: str = ""):
    """produce a heat map for a specific layer across all attention heads each on their own subplot

    Args:
        attention_heads (_type_): a tensor of shape (n_heads, seq_len, seq_len) containing the attention scores for a specific layer across all attention heads
        letters (str): the letters corresponding to the sequence length dimension of the attention heads, used for labeling the axes of the heat map
        layer_name (str, optional): the name of the layer for which the heat map is being produced, used for the title of the heat map. Defaults to "".
    """

    n_heads, seq_len, _ = attention_heads.shape

    #max 3 per row
    n_cols = min(3, n_heads)
    n_rows = (n_heads + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_heads > 1 else [axes]
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

