# train_eval.py (item index 검증 포함)
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from tqdm import tqdm

from src.model.bert4rec import BERT4Rec
from src.model.gatedrec import BERT4RecWithCrossAttention
from src.dataset.datasets import MakeSequenceDataSet, BERTRecDataset, count_unique_items
from utils.config_loader import load_config

config = load_config()
base_cfg = config['base']
train_cfg = config['train']
rec_cfg = config['recommend']
data_cfg = config['data']

sequence_loader = MakeSequenceDataSet(
    data_path=data_cfg['source_path'],
    cross_emb_path=data_cfg['target_cross_emb'],
    spe_emb_path=data_cfg['target_spe_emb']
)
user_train, user_label, user_valid, cross_emb, spe_emb = sequence_loader.get_train_valid_data()
item_min, num_item = count_unique_items(user_train.values())

train_dataset = BERTRecDataset(user_train, base_cfg['max_len'], len(user_train), item_min, num_item, train_cfg['mask_prob'])
data_loader = DataLoader(train_dataset, batch_size=rec_cfg['batch_size'], shuffle=False)

backbone = BERT4Rec(
    num_user=len(user_train),
    num_item=num_item,
    hidden_units=base_cfg['hidden_units'],
    num_heads=base_cfg['num_heads'],
    num_layers=base_cfg['num_layers'],
    max_len=base_cfg['max_len'],
    dropout_rate=base_cfg['dropout_rate'],
    device=base_cfg['device']
)
model = BERT4RecWithCrossAttention(backbone, num_item, base_cfg['hidden_units'], base_cfg['device']).to(base_cfg['device'])

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=rec_cfg['lr'])

labels = list(user_label.values())
valids = list(user_valid.values())
label_batches = [labels[i:i+rec_cfg['batch_size']] for i in range(0, len(labels), rec_cfg['batch_size'])]
valid_batches = [valids[i:i+rec_cfg['batch_size']] for i in range(0, len(valids), rec_cfg['batch_size'])]
cross_batches = [cross_emb[i:i+rec_cfg['batch_size']] for i in range(0, len(cross_emb), rec_cfg['batch_size'])]
spe_batches = [spe_emb[i:i+rec_cfg['batch_size']] for i in range(0, len(spe_emb), rec_cfg['batch_size'])]

def train(model, loader):
    model.train()
    loss_total = 0
    for i, seq in enumerate(loader):
        current_batch_size = seq.size(0)
        spe = torch.tensor(spe_batches[i], dtype=torch.float32).to(base_cfg['device'])[:current_batch_size]
        cross = torch.tensor(cross_batches[i], dtype=torch.float32).to(base_cfg['device'])[:current_batch_size]
        labels = torch.tensor(label_batches[i]).to(base_cfg['device'])[:current_batch_size]
        logits = model(seq.to(base_cfg['device']), spe, cross)

        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    return loss_total / len(loader)

def evaluate():
    model.eval()
    NDCG_5 = NDCG_10 = HIT_1 = HIT_5 = HIT_10 = MRR = 0
    with torch.no_grad():
        for user in range(len(user_train)):
            seq = (user_train[user] + [user_label[user]] + [0])[-base_cfg['max_len']:]
            rated = [user_valid[user]]
            items = [user_valid[user]] + train_dataset.random_neg_sampling(rated, 999)

            seq_tensor = torch.LongTensor([seq]).to(base_cfg['device'])
            spe_tensor = torch.tensor(spe_emb[user], dtype=torch.float32).unsqueeze(0).to(base_cfg['device'])
            cross_tensor = torch.tensor(cross_emb[user], dtype=torch.float32).unsqueeze(0).to(base_cfg['device'])

            pred = -model(seq_tensor, spe_tensor, cross_tensor).squeeze()

            # ✅ item index가 pred 범위 내에 있는지 확인
            items = [item for item in items if item < pred.shape[0]]
            if not items:
                continue

            pred = pred[items]
            rank = pred.argsort().argsort()[0].item()

            if rank < 1: HIT_1 += 1
            if rank < 5: NDCG_5 += 1 / np.log2(rank + 2); HIT_5 += 1
            if rank < 10: NDCG_10 += 1 / np.log2(rank + 2); HIT_10 += 1
            MRR += 1 / (rank + 1)

    N = len(user_train)
    return NDCG_5/N, NDCG_10/N, HIT_1/N, HIT_5/N, HIT_10/N, MRR/N

for epoch in range(1, rec_cfg['num_epochs']+1):
    train_loss = train(model, data_loader)
    ndcg5, ndcg10, hit1, hit5, hit10, mrr = evaluate()
    print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | NDCG@5: {ndcg5:.4f} | HIT@5: {hit5:.4f} | MRR: {mrr:.4f}")

torch.save(model.state_dict(), rec_cfg['checkpoint_output'])