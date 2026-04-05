import torch
import attention
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_create_kqv_matrix():
    # Define arbitrary dimensions for testing
    input_dim = 8
    n_heads = 2
    
    # Create the layer
    layer = attention.create_kqv_matrix(input_vector_dim=input_dim, n_heads=n_heads)
    
    # Verify the object type is a PyTorch Linear layer
    assert isinstance(layer, nn.Linear), "Output must be an nn.Linear layer."
    
    # Verify the input and output dimensions. 
    # Output should be 3x the input dimension to accommodate Q, K, and V.
    assert layer.in_features == input_dim, f"Expected in_features to be {input_dim}."
    assert layer.out_features == input_dim * 3, f"Expected out_features to be {input_dim * 3}."
    
    print("test_create_kqv_matrix passed successfully!")

def test_kqv():
    # Setup dimensions: Batch size=2, Sequence length=3, Embedding dim=4
    B, N, D = 2, 3, 4
    x = torch.rand(B, N, D, device=DEVICE)
    
    # Create a linear layer that matches the dimensions and move it to DEVICE
    linear = nn.Linear(D, D * 3).to(DEVICE)
    
    # Call the function
    k, q, v = attention.kqv(x, linear)
    
    # Verify that the chunking correctly returned 3 tensors of the original shape
    expected_shape = (B, N, D)
    assert k.size() == expected_shape, f"k shape mismatch. Expected {expected_shape}, got {k.size()}"
    assert q.size() == expected_shape, f"q shape mismatch. Expected {expected_shape}, got {q.size()}"
    assert v.size() == expected_shape, f"v shape mismatch. Expected {expected_shape}, got {v.size()}"
    
    print("test_kqv passed successfully!")

def test_attention_scores():
    # Setup test variables and move them to DEVICE
    a = torch.tensor([[
        [ 1.0,  1.0,  1.0,  1.0],  
        [-1.0, -1.0, -1.0, -1.0]   
    ]], device=DEVICE)
    
    b = torch.tensor([[
        [ 2.0,  2.0,  2.0,  2.0],  
        [ 0.0,  1.0,  0.0,  1.0]   
    ]], device=DEVICE)
    
    expected_output = torch.tensor([[
        [ 4.0,  1.0],
        [-4.0, -1.0]
    ]], device=DEVICE)
    
    A = attention.attention_scores(a, b)
    
    assert torch.allclose(A, expected_output), "Test failed! Output does not match expected."
    print("test_attention_scores passed successfully!")

def test_create_causal_mask():
    # Setup test variables
    max_context_len = 4
    
    # Call the function (embed_dim and n_heads are deliberately ignored by the function)
    mask = attention.create_causal_mask(embed_dim=128, n_heads=8, max_context_len=max_context_len)
    
    # Verify shape
    assert mask.size() == (max_context_len, max_context_len), "Mask shape is incorrect."
    
    # Verify the values mathematically and ensure it is on DEVICE for comparison
    expected_mask = torch.tensor([
        [1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.]
    ], device=DEVICE)
    
    assert torch.allclose(mask, expected_mask), "Mask values do not match a causal lower-triangular pattern."
    
    print("test_create_causal_mask passed successfully!")

def test_self_attention():
    # Setup V with easily calculable numbers
    v = torch.tensor([[
        [1.0, 2.0],  # Word 1
        [3.0, 4.0]   # Word 2
    ]], device=DEVICE) # Shape: (1, 2, 2)
    
    # Setup A (Attention scores before softmax). Setting to 0s yields 50/50 probabilities.
    A = torch.tensor([[
        [0.0, 0.0],
        [0.0, 0.0]
    ]], device=DEVICE) # Shape: (1, 2, 2)
    
    # --- Test 1: Without Mask ---
    # Softmax of [0, 0] is [0.5, 0.5].
    # Expected output per word: 0.5 * [1,2] + 0.5 * [3,4] = [2.0, 3.0]
    expected_sa_no_mask = torch.tensor([[
        [2.0, 3.0],
        [2.0, 3.0]
    ]], device=DEVICE)
    
    sa_no_mask = attention.self_attention(v, A, mask=None)
    assert torch.allclose(sa_no_mask, expected_sa_no_mask), "Self-attention without mask failed."
    
    # --- Test 2: With Causal Mask ---
    mask = torch.tensor([
        [1., 0.],
        [1., 1.]
    ], device=DEVICE)
    
    # Word 1 mask is [1, 0]. Softmax([0, -inf]) becomes [1.0, 0.0]
    # Expected word 1: 1.0 * [1,2] + 0.0 * [3,4] = [1.0, 2.0]
    # Word 2 mask is [1, 1]. Softmax([0, 0]) becomes [0.5, 0.5]
    # Expected word 2: 0.5 * [1,2] + 0.5 * [3,4] = [2.0, 3.0]
    expected_sa_with_mask = torch.tensor([[
        [1.0, 2.0],
        [2.0, 3.0]
    ]], device=DEVICE)
    
    sa_with_mask = attention.self_attention(v, A, mask=mask)
    assert torch.allclose(sa_with_mask, expected_sa_with_mask), "Self-attention with mask failed."
    
    print("test_self_attention passed successfully!")

def test_self_attention_layer():
    # Setup random tensors for a forward pass test
    B, N, D = 2, 5, 8
    x = torch.rand(B, N, D, device=DEVICE)
    kqv_matrix = nn.Linear(D, D * 3).to(DEVICE) # Ensure layer is on DEVICE
    mask = torch.tril(torch.ones(N, N, device=DEVICE))
    
    # Call the layer
    sa = attention.self_attention_layer(x, kqv_matrix, mask)
    
    # Verify that the final output shape strictly matches the input shape
    assert sa.size() == x.size(), f"Output shape mismatch. Expected {x.size()}, got {sa.size()}."
    
    print("test_self_attention_layer passed successfully!")

def test_multi_head_attention_layer():
    # Setup random tensors
    B, N, D = 2, 5, 8
    x = torch.rand(B, N, D, device=DEVICE)
    mask = torch.tril(torch.ones(N, N, device=DEVICE))
    
    # If using the basic list implementation, ensure layers inside the list are on DEVICE:
    kqv_matrices = [nn.Linear(D, D * 3).to(DEVICE)] 
    
    # Call the layer
    sa = attention.multi_head_attention_layer(x, kqv_matrices, mask)
    
    # Verify shape consistency
    assert sa.size() == x.size(), f"Output shape mismatch. Expected {x.size()}, got {sa.size()}."
    
    print("test_multi_head_attention_layer passed successfully!")

def test_causal_self_attention():
    # Setup random tensors
    B, N, D = 2, 5, 8
    x = torch.rand(B, N, D, device=DEVICE)
    kqv_matrix = nn.Linear(D, D * 3).to(DEVICE)
    
    # Check if causal_self_attention is implemented in attention.py before testing
    if hasattr(attention, 'causal_self_attention'):
        sa = attention.causal_self_attention(x, kqv_matrix)
        assert sa.size() == x.size(), "Output shape mismatch in causal_self_attention."
        print("test_causal_self_attention passed successfully!")
    else:
        print("causal_self_attention is not implemented yet. Skipping test.")
