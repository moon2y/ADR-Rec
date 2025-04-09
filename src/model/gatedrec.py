# src/model/gatedrec.py 수정 - pooled embedding 사용으로 shape 오류 방지
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionGating(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.linear1 = nn.Linear(1, hidden_units)  # input: attention scalar
        self.linear2 = nn.Linear(1, hidden_units)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pooled, vector1, vector2):  # [B, H], [B, H], [B, H]
        attention_weights1 = torch.sum(pooled * vector1, dim=1, keepdim=True)  # [B, 1]
        attention_weights2 = torch.sum(pooled * vector2, dim=1, keepdim=True)  # [B, 1]

        gate1 = self.sigmoid(self.linear1(attention_weights1))  # [B, H]
        gate2 = self.sigmoid(self.linear2(attention_weights2))  # [B, H]

        gated = gate1 * pooled + gate2 * pooled  # [B, H]
        return gated

class BERT4RecWithCrossAttention(nn.Module):
    def __init__(self, bert4rec_model, num_item, hidden_units, device):
        super().__init__()
        self.bert4rec = bert4rec_model
        self.cross_attention = CrossAttentionGating(hidden_units)
        self.item_embeddings = nn.Embedding(num_item + 1, hidden_units)  # ✅ item 인덱스 범위 보장
        self.device = device

    def forward(self, log_seqs, vector1, vector2):
        _, pooled = self.bert4rec(log_seqs)  # [B, hidden_units]
        combined = self.cross_attention(pooled, vector1, vector2)  # [B, hidden_units]

        item_indices = torch.arange(self.item_embeddings.num_embeddings).to(self.device)
        item_embs = self.item_embeddings(item_indices)  # [num_items, hidden_units]

        logits = torch.matmul(combined, item_embs.transpose(0, 1))  # [B, num_items]
        return logits