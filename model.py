import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from mingpt.utils import CfgNode as CN

class ActivationFunc(nn.Module):
    def forward(self, inp):
        return 0.5 * inp * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (inp + 0.044715 * torch.pow(inp, 3.0))))

class MaskedAttention(nn.Module):
    def __init__(self, params):
        super().__init__()
        assert params.n_embd % params.n_head == 0
        self.qkv_proj = nn.Linear(params.n_embd, 3 * params.n_embd)
        self.out_proj = nn.Linear(params.n_embd, params.n_embd)
        self.dropout_att = nn.Dropout(params.attn_pdrop)
        self.dropout_res = nn.Dropout(params.resid_pdrop)
        self.register_buffer("mask", torch.tril(torch.ones(params.block_size, params.block_size))
                             .view(1, 1, params.block_size, params.block_size))
        self.num_heads = params.n_head
        self.embedding_dim = params.n_embd

    def forward(self, input_tensor):
        B, T, C = input_tensor.size()
        q, k, v = self.qkv_proj(input_tensor).split(self.embedding_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout_att(scores)
        output = scores @ v
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.dropout_res(self.out_proj(output))
        return output

class TransformerLayer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(params.n_embd)
        self.attention_layer = MaskedAttention(params)
        self.layer_norm_2 = nn.LayerNorm(params.n_embd)
        self.feedforward = nn.ModuleDict(dict(
            fc1=nn.Linear(params.n_embd, 4 * params.n_embd),
            fc2=nn.Linear(4 * params.n_embd, params.n_embd),
            activation=ActivationFunc(),
            dropout=nn.Dropout(params.resid_pdrop),
        ))
        self.ffwd = lambda x: self.feedforward.dropout(
            self.feedforward.fc2(self.feedforward.activation(self.feedforward.fc1(x)))
        )

    def forward(self, input_tensor):
        temp = input_tensor + self.attention_layer(self.layer_norm_1(input_tensor))
        return temp + self.ffwd(self.layer_norm_2(temp))

class TransformerModel(nn.Module):
    @staticmethod
    def default_cfg():
        config = CN()
        config.model_variant = 'transformer'
        config.num_layers = None
        config.num_heads = None
        config.embedding_dim = None
        config.vocab_size = None
        config.sequence_length = None
        config.embd_dropout = 0.1
        config.resid_dropout = 0.1
        config.attn_dropout = 0.1
        return config

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.seq_len = config.sequence_length
        self.model_params = nn.ModuleDict(dict(
            token_emb=nn.Embedding(config.vocab_size, config.embedding_dim),
            pos_emb=nn.Embedding(config.sequence_length, config.embedding_dim),
            dropout=nn.Dropout(config.embd_dropout),
            layers=nn.ModuleList([TransformerLayer(config) for _ in range(config.num_layers)]),
            norm_layer=nn.LayerNorm(config.embedding_dim),
        ))
        self.output_proj = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_single_weight)
        for name, param in self.named_parameters():
            if name.endswith('fc2.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * self.seq_len))

    def _init_single_weight(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, variant):
        assert variant in {'transformer-small', 'transformer-medium', 'transformer-large'}
        from transformers import GPT2LMHeadModel
        config = cls.default_cfg()
        config.model_variant = variant
        config.vocab_size = 50257
        config.sequence_length = 1024
        model = TransformerModel(config)
        model_hf = GPT2LMHeadModel.from_pretrained(variant)
        hf_params = model_hf.state_dict()
        model_params = model.state_dict()
        for name in hf_params:
            if any(name.endswith(t) for t in ['fc1.weight', 'fc2.weight']):
                assert hf_params[name].shape[::-1] == model_params[name].shape
                with torch.no_grad():
                    model_params[name].copy_(hf_params[name].t())
            else:
                assert hf_params[name].shape == model_params[name].shape
                with torch.no_grad():
                    model_params[name].copy_(hf_params[name])
        return model
