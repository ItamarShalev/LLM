from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from transformers import loss

def batch_to_labeled_samples(batch: torch.IntTensor) -> tuple[torch.Tensor, torch.Tensor]:
    # DONE implement this.
    # The batches that we get from the reader have corpus-sequences of length max-context + 1.
    # We need to translate them to input/output examples, each of which is shorter by one.
    # That is, if our input is of dimension (b x n) our output is two tensors, each of dimension (b x n-1)
    inputs = batch[:, :-1] # DONE fix this
    labels = batch[:, 1:] # DONE fix this
    return (inputs, labels)

def compute_loss(logits, gold_labels):
    # logits size is (batch, seq_len, vocab_size)
    # gold_bales size is (batch, seq_len)
    # NOTE remember to handle padding (ignore them in loss calculation!)
    # NOTE cross-entropy expects other dimensions for logits
    # NOTE you can either use cross_entropy from PyTorch, or implement the loss on your own.
    logits_flat = logits.view(-1, logits.size(-1)) # (batch * seq_len, vocab_size)
    labels_flat = gold_labels.view(-1) # (batch * seq_len)
    loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=0)
    return loss
