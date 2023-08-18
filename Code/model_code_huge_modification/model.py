import copy
from typing import Optional, Any, Union, Callable
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
from torch.nn import MultiheadAttention
import torch.nn as nn
import numpy as np
import math

# Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale, dropout_rate):
        super(ScaledDotProductAttention, self).__init__()

        self.scale = scale # match.sqrt(d_k)
        self.dropout_rate = dropout_rate

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask):
        # query: (B, n_heads, query_len, head_dim)
        # key: (B, n_heads, key_len, head_dim)
        # value: (B, n_heads, value_len, head_dim)
        # mask: (B, 1, 1, source_seq_len) - for sequence mask
        #       (B, 1, target_seq_len, target_seq_len) - for target mask

        # calculate alignment scores
        scores = torch.matmul(query, key.transpose(-2, -1)) #(B, n_heads, query_len, key_len)
        scores = scores / self.scale #(B, n_heads, query_len, key_len)
        
        # mask out invalid positions
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf')) #(B, n_heads, query_len, key_len)
    
        # calculate the attention weights (prob)
        attn_probs = F.softmax(scores, dim = -1)
        
        # calculate context vector
        output = torch.matmul(self.dropout(attn_probs), value) #(B, n_heads, query_len, head_dim)

        # output: (B, n_heads, query_len, head_dim)
        # attn_probs: (B, n_heads, query_len, value_len)
        return output, attn_probs

# Multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "'d_model' should be a multiple of 'n_heads'"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads # head_dim
        self.dropout_rate = dropout_rate

        self.W_q = nn.Linear(d_model, d_model, bias = False)
        self.W_k = nn.Linear(d_model, d_model, bias = False)
        self.W_v = nn.Linear(d_model, d_model, bias = False)
        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(np.sqrt(self.d_k), dropout_rate=dropout_rate)

    def split_heads(self, x):
        # x: (B, seq_len, d_model)
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # x: (B, n_heads, seq_len, d_k = head_dim)
        return x
    
    def group_heads(self, x):
        # x: (B, n_heads, seq_len, d_k = head_dim)
        batch_size = x.size(0)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)

        # x: (B, seq_len, d_model)
        return x
    
    def forward(self, query, key, value, mask = None):
        '''
            args:
                query: (B, query_len, d_model)
                key: (B, key_len, d_model)
                value: (B, value_len, d_model)
                mask: (B, 1, source_seq_len) for source mask
                    (B, target_seq_len, target_seq_len) for target mask
        '''
        Q = self.split_heads(self.W_q(query)) # (B, n_heads, query_len, head_dim)
        K = self.split_heads(self.W_k(key)) # (B, n_heads, key_len, head_dim)
        V = self.split_heads(self.W_v(value)) # (B, n_heads, value_len, head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1)
        
        x, attn = self.attention(Q, K, V, mask)
        
        x = self.group_heads(x)

        x = self.W_o(x)

        return x, attn

# Position-wise Feed-Forward Network
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (B, seq_len, d_model)

        x = self.dropout(F.relu(self.w_1(x))) # (B, seq_len, d_ff)
        x = self.w_2(x) # (B, seq_len, d_model

        return x

# Positional Encoding Module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate = 0.1, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        self.dropout = nn.Dropout(dropout_rate)

        # compute positional encodings
        pe = torch.zeros(max_len, d_model) #(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #(max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        ) # (d_model, )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, seq_len, d_model)
        x = x + self.pe[:x.size(0), :] # (B, seq_len, d_model)
        x = self.dropout(x) # (B, seq_len, d_model)

        return x
    
# EncoderLayer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate = 0.1, **kwargs):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        if 'attn_layer' in kwargs.keys():
            self.attn_layer = kwargs['attn_layer']
        else:
            self.attn_layer = MultiHeadAttention(d_model, n_heads, dropout_rate)
        
        self.attn_layer_norm = nn.LayerNorm(d_model, eps = 1e-6)

        if 'ff_layer' in kwargs.keys():
            self.ff_layer = kwargs['ff_layer']
        else:
            self.ff_layer = PositionwiseFeedForward(d_model, d_ff, dropout_rate)

        self.ff_layer_norm = nn.LayerNorm(d_model, eps = 1e-6)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        '''
            args:
                x: (B, source_seq_len, d_model)
                mask: (B, 1, source_seq_len)
        '''
        # apply self attention
        x1, _ = self.attn_layer(x, x, x, mask) # (B, source_seq_len, d_model)

        # apply residual connection followed by layer normalization
        x = self.attn_layer_norm(x + self.dropout(x1))

        # apply position-wise feed-forward
        x1 = self.ff_layer(x) # (B, soure_seq_len, d_model)

        # apply residual connection followd by layer normalization
        x = self.ff_layer_norm(x + self.dropout(x1)) # (B, source_seq_len, d_model)

        # x: (B, source_seq_len, d_model)
        return x

# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, pad_idx, 
                 dropout_rate=0.1, max_len = 5000, **kwargs):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.pad_idx = pad_idx
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        # tok embedding
        if 'tok_embedding' in kwargs.keys():
            self.tok_embedding = kwargs['tok_embedding']
        else:
            self.tok_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # pos embedding
        self.pos_embedding = PositionalEncoding(d_model, dropout_rate, max_len)

        # encoder layers
        if 'layers' in kwargs.keys():
            self.layers = kwargs['layers']
        else:
            self.layers = nn.ModuleList([
                EncoderLayer(d_model, n_heads, d_ff, dropout_rate)
                for _ in range(n_layers)
            ])
        
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, x, mask):
        # x: (B, source_seq_len)
        # mask: (B, 1, source_seq_len)

        x = self.tok_embedding(x)
        x = self.pos_embedding(x)

        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.layer_norm(x)

        return x
    
# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1, **kwargs):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

        if 'attn_layer' in kwargs.keys():
            self.attn_layer = kwargs['attn_layer']
        else:    
            self.attn_layer = MultiHeadAttention(d_model, n_heads, dropout_rate)
        
        self.attn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        if 'enc_attn_layer' in kwargs.keys():
            self.enc_attn_layer = kwargs['enc_attn_layer']
        else:
            self.enc_attn_layer = MultiHeadAttention(d_model, n_heads, dropout_rate)
        
        self.enc_attn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        if 'ff_layer' in kwargs.keys():
            self.ff_layer = kwargs['ff_layer']
        else:
            self.ff_layer = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        
        self.ff_layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, memory, src_mask, tgt_mask):
        # apply self attention
        x1, _ = self.attn_layer(x, x, x, tgt_mask) # (B, taret_seq_len, d_model)

        # apply residual connection followed by layer normalization
        x = self.attn_layer_norm(x + self.dropout(x1)) # (B, target_seq_len, d_model)

        # apply encoder-decoder attention
        # memory is the output from encoder block
        x1, attn = self.enc_attn_layer(x, memory, memory, src_mask) 
            # x1: (B, target_seq_len, d_model)
            # attn: (B, n_heads, target_seq_len, source_seq_len)
        
        # apply residual connection followed by layer normalization
        x = self.enc_attn_layer_norm(x + self.dropout(x1)) # (B, target_seq_len, d_model)

        # apply position-wise feed-forward
        x1 = self.ff_layer(x) # (B, target_seq_len, d_model)

        # apply residual conneciton followed by layer normalization
        x = self.ff_layer_norm(x + self.dropout(x1)) # (B, taret_seq_len, d_model)

        return x, attn

# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, pad_idx, 
                 dropout_rate=0.1, max_len=5000, **kwargs):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.pad_idx = pad_idx
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        if 'tok_embedding' in kwargs.keys():
            self.tok_embedding = kwargs['tok_embedding']
        else:
            self.tok_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        self.pos_embedding = PositionalEncoding(d_model, dropout_rate, max_len)

        if 'layers' in kwargs.keys():
            self.layers = kwargs['layers']
        else:
            self.layers = nn.ModuleList([
                DecoderLayer(d_model, n_heads, d_ff, dropout_rate)
                for _ in range(n_layers)
            ])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.tok_embedding(x) # (B, target_seq_len, d_model)
        x = self.pos_embedding(x) # (B, target_seq_len, d_model)

        for layer in self.layers:
            x, attn = layer(x, memory, src_mask, tgt_mask)
        
        x = self.layer_norm(x)

        return x, attn        

# Transformer -> wrapper for encoder and decoder
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, generator, pad_idx):
        super(Transformer, self).__init__()
        self.pad_idx = pad_idx
        
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def get_pad_mask(self, x, pad_idx):
        # x: (B, seq_len)
        x = (x != pad_idx).unsqueeze(-2) # (B, 1, seq_len)

        return x
    
    def get_subsquent_mask(self, x):
        # x: (B, seq_len)
        seq_len = x.size(1)
        subsquent_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1).astype(np.int8) # (B, seq_len, seq_len)
        subsquent_mask = (torch.from_numpy(subsquent_mask) == 0).to(x.device) # load to device

        return subsquent_mask
    
    def forward(self, src, tgt):
        '''
            args:
                src: (B, soruce_seq_len)
                tgt: (B, target_seq_len)
        '''
        # create masks for source and target
        src_mask = self.get_pad_mask(src, self.pad_idx) # (B, 1, seq_len)
        tgt_mask = self.get_pad_mask(tgt, self.pad_idx) & self.get_subsquent_mask(tgt) # (B, seq_len, seq_len)

        # encode the source sequence
        enc_output = self.encoder(src, src_mask) # (B, source_seq_len, d_model)

        # decode based on source sequence and target sequence generated so far
        dec_output, attn = self.decoder(tgt, enc_output, src_mask, tgt_mask)
            # dec_output: (B, target_seq_len, d_model)
            # attn: (B, n_heads, target_seq_len, source_seq_len)
        
        # apply linear projection to obtain the output distribution
        output = self.generator(dec_output) # (B, target_seq_len, vocab_size)

        return output, attn

# Generator - Linear projection layer for generating output distribution
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (B, target_seq_len, d_model)
        
        # apply linear projection followed by softmax to obtain output distribution
        x = self.proj(x) # (B, target_seq_len, vocab_size)
        output = F.log_softmax(x, dim=-1) # (B, target_seq_len, vocab_size)

        return output

class ProposedModel(nn.Module):
    def __init__(self, transformer1, transformer2):
        super(ProposedModel, self).__init__()

        self.transformer1 = transformer1
        self.transformer2 = transformer2

    def forward(self, lan1: Tensor, lan2: Tensor, pad_idx, eos_idx, loop_cond):
        
        if loop_cond == 'start1_from_lan1_to_lan2':
            # lan1 -> lan2
            '''
                args: lan1, lan2
                    lan1 - (batch, lan1_seq_len) - encoder input
                    lan2 - (batch, lan2_seq_len) - decoder input
                    --------------------------------------------
                    encoder input -> [SOS] t1 t2 ... tm [EOS] [PAD] [PAD] ...
                    decoder input -> [SOS] k1 k2 ... kn [PAD] [PAD] [PAD] ...
                results:
                    output - (batch, lan2_seq_len, vocab_size)
            '''
            ## decoder input
            tgt_clone = lan2.clone().detach()
            tgt_clone[lan2 == eos_idx] = pad_idx

            output, _ = self.transformer1(lan1, tgt_clone[:, :-1], pad_idx, eos_idx)

        elif loop_cond == 'start1_from_lan2_to_lan1':
            # lan2 -> lan1
            '''
                args: lan2, lan1
                    lan2 - (batch, lan2_seq_len) - encoder input
                    lan1 - (batch, lan1_seq_len) - decoder input
                    ---------------------------------------------
                    encoder input -> [SOS] t1 t2 ... tm [EOS] [PAD] [PAD] ...
                    decoder input -> [SOS] k1 k2 ... kn [PAD] [PAD] [PAD] ...
                results:
                    output - (batch, lan1_seq_len, vocab_size)
            '''
            ## decoder input
            tgt_clone = lan1.clone().detach()
            tgt_clone[lan1 == eos_idx] = pad_idx

            output, _ = self.transformer2(lan2, tgt_clone[:, :-1], pad_idx, eos_idx)

        elif loop_cond == 'start2_from_lan2_to_lan1':
            # lan2 -> lan1
            '''
                args: lan2, lan1
                    lan2 - (batch, lan2_seq_len) - encoder input
                    lan1 - (batch, lan1_seq_len) - decoder input
                    ---------------------------------------------
                    encoder input -> [SOS] t1 t2 ... tm [EOS] [PAD] [PAD] ...
                    decoder input -> [SOS] k1 k2 ... kn [PAD] [PAD] [PAD] ...
                results:
                    output - (batch, lan1_seq_len, vocab_size)
            '''
            ## decoder input
            tgt_clone = lan2.clone().detach()
            tgt_clone[lan2 == eos_idx] = pad_idx

            output, _ = self.transformer2(lan2, tgt_clone[:, :-1], pad_idx, eos_idx)

        elif loop_cond == 'start_2_from_lan1_to_lan2':
            # lan1 -> lan2
            '''
                args: lan1, lan2
                    lan1 - (batch, lan1_seq_len) - encoder input
                    lan2 - (batch, lan2_seq_len) - decoder input
                    ---------------------------------------------
                    encoder input -> [SOS] t1 t2 ... tm [EOS] [PAD] [PAD] ...
                    decoder input -> [SOS] k1 k2 ... kn [PAD] [PAD] [PAD] ...
                results:
                    output - (batch, lan2_seq_len, d_model, vocab_size)
            '''
            ## decoder input
            tgt_clone = lan1.clone().detach()
            tgt_clone[lan1 == eos_idx] = pad_idx

            output, _ = self.transformer1(lan1, tgt_clone[:, :-1], pad_idx, eos_idx)

        return output