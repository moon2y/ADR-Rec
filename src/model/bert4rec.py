import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, Q, K, V, mask):
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.hidden_units)
        attn_score = attn_score.masked_fill(mask == 0, -1e9)
        attn_dist = self.dropout(F.softmax(attn_score, dim=-1))
        output = torch.matmul(attn_dist, V)
        return output, attn_dist

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_units = hidden_units

        self.W_Q = nn.Linear(hidden_units, hidden_units * num_heads, bias=False)
        self.W_K = nn.Linear(hidden_units, hidden_units * num_heads, bias=False)
        self.W_V = nn.Linear(hidden_units, hidden_units * num_heads, bias=False)
        self.W_O = nn.Linear(hidden_units * num_heads, hidden_units, bias=False)

        self.attention = ScaledDotProductAttention(hidden_units, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.layerNorm = nn.LayerNorm(hidden_units, eps=1e-6)

    def forward(self, enc, mask):
        residual = enc
        batch_size, seqlen = enc.size(0), enc.size(1)

        Q = self.W_Q(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units).transpose(1, 2)
        K = self.W_K(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units).transpose(1, 2)
        V = self.W_V(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units).transpose(1, 2)

        output, attn_dist = self.attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, seqlen, -1)
        output = self.layerNorm(self.dropout(self.W_O(output)) + residual)
        return output, attn_dist

class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.W_1 = nn.Linear(hidden_units, hidden_units)
        self.W_2 = nn.Linear(hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.layerNorm = nn.LayerNorm(hidden_units, eps=1e-6)

    def forward(self, x):
        residual = x
        output = self.W_2(F.relu(self.dropout(self.W_1(x))))
        return self.layerNorm(self.dropout(output) + residual)

class BERT4RecBlock(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, hidden_units, dropout_rate)
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_units, dropout_rate)

    def forward(self, x, mask):
        x, _ = self.attention(x, mask)
        x = self.pointwise_feedforward(x)
        return x

class BERT4Rec(nn.Module):
    def __init__(self, num_user, num_item, hidden_units, num_heads, num_layers, max_len, dropout_rate, device):
        super().__init__()
        self.item_emb = nn.Embedding(num_item + 2, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.emb_layernorm = nn.LayerNorm(hidden_units, eps=1e-6)
        self.blocks = nn.ModuleList([
            BERT4RecBlock(num_heads, hidden_units, dropout_rate) for _ in range(num_layers)
        ])
        self.device = device
        self.dout = nn.Linear(hidden_units, 50)
        self.bout = nn.Linear(hidden_units, num_item + 1)

    def forward(self, log_seqs):
        log_seqs = log_seqs.to(self.device)
        seqs = self.item_emb(log_seqs)
        positions = torch.arange(log_seqs.shape[1]).unsqueeze(0).expand(log_seqs.shape[0], -1).to(self.device)
        seqs += self.pos_emb(positions)
        seqs = self.emb_layernorm(self.dropout(seqs))

        mask = (log_seqs > 0).unsqueeze(1).repeat(1, log_seqs.shape[1], 1).unsqueeze(1)
        for block in self.blocks:
            seqs = block(seqs, mask)

        pooled_out = torch.mean(self.dout(seqs), dim=1)
        return self.bout(seqs), pooled_out