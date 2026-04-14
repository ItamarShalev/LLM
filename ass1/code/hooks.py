"""The purpose of this file is to define hooks for extracting tensors from the Transformers model
"""

import torch


ATTENTION_HEADS = {}

def attention_hook(name: str):
    """Defines a hook to extract an attention layer

    Args:
        name (str): The name we will give to the attention layer
    """
    def hook(model: torch.nn.Module, input: tuple, output: torch.Tensor):
        # input is a tuple, and we want the first element of 
        ATTENTION_HEADS[name] = output.detach().cpu()
    return hook

