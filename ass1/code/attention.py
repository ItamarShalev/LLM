from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
import math


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_kqv_matrix(input_vector_dim, n_heads = 1):
    output_vector_dim = input_vector_dim * 3 // n_heads
    return nn.Linear(input_vector_dim, output_vector_dim) # DONE fill in the correct dimensions

def kqv(x, linear):
    B, N, D = x.size()
    # DONE compute k, q, and v
    kqv_out = linear(x)
    k, q, v = kqv_out.chunk(3, dim=-1)
    # (can do it in 1 or 2 lines.)
    return k, q, v

def attention_scores(a, b):

    B1, N1, D1 = a.size()
    B2, N2, D2 = b.size()
    assert B1 == B2
    assert D1 == D2

    # DONE compute A (remember: we are computing *scaled* dot product attention. don't forget the scaling.
    # (can do it in 1 or 2 lines.)
    A = (a @ b.transpose(-2, -1)) / math.sqrt(D1)
    return A

def create_causal_mask(embed_dim, n_heads, max_context_len):
    # Return a causal mask (a tensor) with zeroes in dimensions we want to zero out.
    # This function receives more arguments than it actually needs. This is just because
    # it is part of an assignment, and I want you to figure out on your own which arguments
    # are relevant.
    mask = torch.ones(max_context_len, max_context_len, device=DEVICE)
    mask = torch.tril(mask) # DONE replace this line with the creation of a causal mask.
    return mask

def self_attention(v, A, mask = None):
    # DONE compute sa (corresponding to y in the assignemnt text).
    # This should take very few lines of code.
    # As usual, the dimensions of v and of sa are (b x n x d).
    if mask is not None:
        A = A.masked_fill(mask == 0, float('-inf'))
    attention_weights = torch.softmax(A, dim=-1)
    sa = attention_weights @ v
    return sa


def self_attention_layer(x, kqv_matrix, attention_mask):
    k, q, v = kqv(x, kqv_matrix)
    att = attention_scores(k, q)
    sa = self_attention(v, att, attention_mask)
    return sa

def multi_head_attention_layer(x, kqv_matrices, mask):
    # DONE implement multi-head attention.
    # This is most easily done using calls to self_attention_layer, each with a different
    # entry in kqv_matrices, and combining the results.
    #
    # There is also a tricker (but more efficient) version of multi-head attention, where we do all the computation
    # using a single multiplication with a single kqv_matrix (or a single kqv_tensor) and re-arranging the results afterwards.
    # If you want a challenge, you can try and implement this. You may need to change additional places in the code accordingly.
    B, N, D = x.size()
    sa_heads = []
    for kqv_matrix in kqv_matrices:
        sa_head = self_attention_layer(x, kqv_matrix, mask)
        sa_heads.append(sa_head)
    sa = torch.cat(sa_heads, dim=-1)

    assert sa.size() == x.size()
    return sa

def create_kqv_matrix_efficient(input_vector_dim: int, n_heads: int = 1) -> torch.Tensor:
    """Implementation of effecient kqv matrix generation given heads, returned tensor is of size (n_head, input_vector_dim, input_vector_dim*3/n_heads)

    Args:
        input_vector_dim (int): the input dimension
        n_heads (int, optional): the number of heads. Defaults to 1.

    Returns:
        torch.Tensor: the kqv matrix for efficient multi-head attention, of size (n_head, input_vector_dim, input_vector_dim*3/n_heads)
    """
    return torch.zeros((n_heads, input_vector_dim, input_vector_dim * 3 // n_heads))

def attention_scores_efficient(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Implementation of efficient attention score calculation for multi-head attention, where a and b are of size (B, H, N, D_head) and the output is of size (B, H, N, N)

    Args:
        a (torch.Tensor): the first input tensor, of size (B, H, N, D_head)
        b (torch.Tensor): the second input tensor, of size (B, H, N, D_head)

    Returns:
        torch.Tensor: the attention scores, of size (B, H, N, N)
    """
    B1, H1, N1, D1 = a.size()
    B2, H2, N2, D2 = b.size()
    assert B1 == B2
    assert H1 == H2
    assert D1 == D2
    D = a.size(-1)
    # DONE compute A (remember: we are computing *scaled* dot product attention. don't forget the scaling.
    # (can do it in 1 or 2 lines.)
    A = (a @ b.transpose(-2, -1)) / math.sqrt(D) #batch matrix multiplication, with scaling
    return A

def multi_head_attention_layer_efficient(x, kqv_tensor, kqv_bias, mask):
    B, N, D = x.size()

    n_heads = kqv_tensor.size(0)
    
    # compute k, q, v for all heads in a single operation, using kqv_tensor and kqv_bias.
    kqv_out = torch.einsum('bnd, hdf -> bhnf', x, kqv_tensor)
    
    # Add the bias term to kqv_out.
    kqv_out = kqv_out + kqv_bias.view(1, n_heads, 1, -1)
    
    # Now split kqv_out into k, q, and v. The resulting tensors should each be of shape (B, H, N, D_head), where D_head = D // n_heads.
    k, q, v = kqv_out.chunk(3, dim=-1)
    
    # Compute attention scores using the efficient attention_scores_efficient function.
    att = attention_scores_efficient(k, q)

    # Compute the self-attention output using the self_attention function.
    sa = self_attention(v, att, mask)
  
    # Finally, we need to combine the outputs of the different heads back into a single tensor of shape (B, N, D).
    sa = sa.transpose(1, 2).contiguous().view(B, N, D)

    assert sa.size() == x.size()

    return sa
    


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_context_len, efficient = False):
        super().__init__()
        assert embed_dim % n_heads == 0
        # the linear layers used for k, q, v computations:
        # each linear is for a different head, but for all of k, q and v for this head.
        self.efficient = efficient
        if efficient:
            self.kqv_matrices = nn.Parameter(create_kqv_matrix_efficient(embed_dim, n_heads))
            self.kqv_bias = nn.Parameter(torch.zeros(n_heads, 3 * embed_dim // n_heads))
        else:
            self.kqv_matrices = nn.ModuleList([create_kqv_matrix(embed_dim, n_heads) for _ in range(n_heads)])
        # For use in the causal part.  "register_buffer" is used to store a tensor which is fixed but is not a parameter of the model.
        # You can then access it with: self.mask
        mask = create_causal_mask(embed_dim, n_heads, max_context_len)
        self.register_buffer("mask", mask)
        self.n_heads = n_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        seq_len = x.size(1)
        cur_mask = self.mask[:seq_len, :seq_len]
        if self.efficient:
            sa = multi_head_attention_layer_efficient(x, self.kqv_matrices, self.kqv_bias, cur_mask)
        else:
            sa = multi_head_attention_layer(x, self.kqv_matrices, cur_mask)
        return sa
