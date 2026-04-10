import re
from torch import nn
import torch
import torch.nn.functional as F
import attention
import mlp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, mlp_hidden_size: int, max_context_len, with_residuals: bool = False, pre_norm: bool = True, efficient: bool = False):
        super().__init__()
        self.causal_attention = attention.CausalSelfAttention(embed_size, n_heads, max_context_len, efficient)
        self.dropout = nn.Dropout(0.1)
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.with_residuals = with_residuals
        self.pre_norm = pre_norm

    def _pre_norm_forward(self, inputs):
        if self.with_residuals:
            # DONE add residuals support.
            attention_out = self.causal_attention(self.layer_norm_1(inputs))
            x = self.dropout(attention_out)
            x = inputs + attention_out
            mlp_out = self.mlp(self.layer_norm_2(x))
            x = x + mlp_out
        else:
            x = inputs
            x = self.layer_norm_1(x)
            x = self.causal_attention(x)
            x = self.dropout(x)
            x = self.layer_norm_2(x)
            x = self.mlp(x)
        return x
    
    def _post_norm_forward(self, inputs):
        if self.with_residuals:
            # DONE add residuals support.
            attention_out = self.causal_attention(inputs)
            attention_out = self.dropout(attention_out)
            x = inputs + attention_out
            x = self.layer_norm_1(x)
            mlp_out = self.mlp(x)
            x = x + mlp_out
            x = self.layer_norm_2(x)
        else:
            x = inputs
            x = self.causal_attention(x)
            x =self.dropout(x)
            x = self.layer_norm_1(x)
            x = self.mlp(x)
            x = self.layer_norm_2(x)
        return x

    def forward(self, inputs):
        if self.pre_norm:
            x = self._pre_norm_forward(inputs)
        else:
            x = self._post_norm_forward(inputs)
        return x

class Embed(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_context_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_context_len, embed_size)
        self.max_context_len = max_context_len

    def forward(self, x):
        # x has the shape (b x n) where b is batch dimension and n is sequence length.
        # each item is an int, indicating a vocabulary item.
        # The output should be of shape (b x n x d), where d is the embedding dimension.
        #tok_embeddings = 
        #pos_embeddings = ...
        b, n = x.size()
        pos_indices = torch.arange(n, device=DEVICE)
        tok_embeddings = self.token_embeddings(x)
        pos_embeddings = self.position_embeddings(pos_indices)
        return tok_embeddings + pos_embeddings


class TransformerLM(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            embed_size: int,
            max_context_len: int,
            vocab_size: int,
            mlp_hidden_size: int,
            with_residuals: bool,
            efficient: bool = False
            ):
        super().__init__()
        self.embed = Embed(vocab_size, embed_size, max_context_len)
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([TransformerDecoderBlock(n_heads, embed_size, mlp_hidden_size, max_context_len, with_residuals, pre_norm=True, efficient=efficient) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embed_size)
        self.word_prediction = nn.Linear(embed_size, vocab_size)
        self.max_context_len = max_context_len

        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print("Parameter count: %.2fM" % (n_params/1e6,))

    def forward(self, inputs):
        x = self.embed(inputs)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        logits = self.word_prediction(x)
        return logits

    def init_weights(self):
        # initialize weights
        # DONE implement initialization logic for embeddings and linear layers.
        # The code break down the parameters by type (layer-norm, linear, embedding),
        # but can also condition on individual names, for example by checking pn.endswith(...).
        for pn, p in self.named_parameters():
            if isinstance(p, nn.LayerNorm):
                torch.nn.init.zeros_(p.bias)
                torch.nn.init.ones_(p.weight)
            elif isinstance(p, nn.Linear):
                # DONE initialize p.weight and p.bias (if it is not None).
                # You can look at initializers in torch.nn.init
                torch.nn.init.kaiming_normal_(p.weight, mode='fan_in', nonlinearity='relu')
                if p.bias is not None:
                    torch.nn.init.zeros_(p.bias)
            elif isinstance(p, nn.Embedding):
                # DONE initialize p.weight and p.bias (if it is not None).
                # You can look at initializers in torch.nn.init
                torch.nn.init.normal_(p.weight, mean=0.0, std=0.02)
            elif 'kqv_matrices' in pn:
                torch.nn.init.normal_(p, mean=0.0, std=0.02)
                

    def sample_continuation(self, prefix: list[int], max_tokens_to_generate: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.int32, device=DEVICE))
                logits_for_last_token = logits[0][-1]
                distribution_for_last_token = F.softmax(logits_for_last_token)
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1)
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated

    def better_sample_continuation(self, prefix: list[int], max_tokens_to_generate: int, temperature: float, topK: int, interpret: bool = False, var: list = None) -> list[int]:
        # TODO implement this.
        # Temperature should be the temperature in which you sample.
        # TopK indicates that we don't sample from the entire distribution, but only from the top k scoring tokens
        # for the given position.
        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.int32, device=DEVICE), interpret, var)
                logits_for_last_token = logits[0][-1]
                # DONE implement temperature and topK sampling.
                distribution_for_last_token = F.softmax(logits_for_last_token / temperature)
                topk_distribution, topk_indices = torch.topk(distribution_for_last_token, topK)
                sampled_index_in_topk = torch.multinomial(topk_distribution, num_samples=1)
                sampled_token = topk_indices[sampled_index_in_topk]
                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)
        return generated
