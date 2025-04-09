import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
import pickle
from src.model.bert4rec import BERT4Rec
from src.model.dann import DANN_BERT4Rec, DANN_Loss
from src.dataset.datasets import BERTRecDataset, count_unique_items
from utils.config_loader import load_config

config = load_config()
base_cfg = config['base']
train_cfg = config['train']
data_cfg = config['data']

with open(data_cfg['target_path'], 'rb') as f:
    domain_A_train = pickle.load(f)
with open(data_cfg['source_path'], 'rb') as f:
    domain_B_train = pickle.load(f)
with open(data_cfg['cross_path'], 'rb') as f:
    domain_C_train = pickle.load(f)

_, num_item = count_unique_items(domain_C_train.values())

train_dataset_A = BERTRecDataset(domain_A_train, base_cfg['max_len'], len(domain_A_train), 1, num_item, train_cfg['mask_prob'])
train_dataset_B = BERTRecDataset(domain_B_train, base_cfg['max_len'], len(domain_B_train), 1, num_item, train_cfg['mask_prob'])
train_dataset_C = BERTRecDataset(domain_C_train, base_cfg['max_len'], len(domain_C_train), 1, num_item, train_cfg['mask_prob'])

data_loader_A = DataLoader(train_dataset_A, batch_size=train_cfg['batch_size'], shuffle=True, pin_memory=True, num_workers=2)
data_loader_B = DataLoader(train_dataset_B, batch_size=train_cfg['batch_size'], shuffle=True, pin_memory=True, num_workers=2)
data_loader_C = DataLoader(train_dataset_C, batch_size=train_cfg['batch_size'], shuffle=True, pin_memory=True, num_workers=2)

base_model = BERT4Rec(
    num_user=len(domain_A_train),
    num_item=num_item,
    hidden_units=base_cfg['hidden_units'],
    num_heads=base_cfg['num_heads'],
    num_layers=base_cfg['num_layers'],
    max_len=base_cfg['max_len'],
    dropout_rate=base_cfg['dropout_rate'],
    device=base_cfg['device'],
)
model = DANN_BERT4Rec(base_model).to(base_cfg['device'])
loss_fn = DANN_Loss().to(base_cfg['device'])

optimizer = AdamW(model.parameters(), lr=train_cfg['lr'])
scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=train_cfg['num_epochs'] * len(data_loader_A))

model.train()
for epoch in range(1, train_cfg['num_epochs'] + 1):
    total_loss_s = 0
    total_loss_c = 0

    for step, (batch_A, batch_B, batch_C) in tqdm(enumerate(zip(data_loader_A, data_loader_B, data_loader_C)), total=len(data_loader_A)):
        x_A, y_A = batch_A.to(base_cfg['device']), batch_A.to(base_cfg['device'])
        x_B, y_B = batch_B.to(base_cfg['device']), batch_B.to(base_cfg['device'])
        x_C, y_C = batch_C.to(base_cfg['device']), batch_C.to(base_cfg['device'])

        result_A = model(x_A)
        result_B = model(x_B)
        result_C = model(x_C)

        loss_A_s, loss_A_c = loss_fn(result_A, y_A, domain_id=0, alpha=train_cfg['alpha'])
        loss_B_s, loss_B_c = loss_fn(result_B, y_B, domain_id=1, alpha=train_cfg['alpha'])
        loss_C_s, loss_C_c = loss_fn(result_C, y_C, domain_id=2, alpha=train_cfg['alpha'])

        loss_s = loss_A_s + loss_B_s + loss_C_s
        loss_c = loss_A_c + loss_B_c + loss_C_c

        optimizer.zero_grad()
        loss_s.backward(retain_graph=True)
        loss_c.backward()
        optimizer.step()
        scheduler.step()

        total_loss_s += loss_s.item()
        total_loss_c += loss_c.item()

    print(f"Epoch {epoch:03d} | Loss_s: {total_loss_s:.4f} | Loss_c: {total_loss_c:.4f}")

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"checkpoints/B4RDANN_epoch{epoch}.pt")
