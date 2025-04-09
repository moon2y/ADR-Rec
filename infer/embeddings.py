import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
import os

from src.model.bert4rec import BERT4Rec
from src.model.dann import DANN_BERT4Rec
from src.dataset.datasets import BERTRecDataset, count_unique_items
from utils.config_loader import load_config

# Load config
config = load_config()
base_cfg = config['base']
infer_cfg = config['infer']
data_cfg = config['data']

# Load data
with open(data_cfg['source_path'], 'rb') as f:
    domain_data = pickle.load(f)

_, num_item = count_unique_items(domain_data.values())

# Dataset
dataset = BERTRecDataset(
    user_train=domain_data,
    max_len=base_cfg['max_len'],
    num_user=len(domain_data),
    item_min=1,
    num_item=num_item,
    mask_prob=config['train']['mask_prob']
)
data_loader = DataLoader(dataset, batch_size=infer_cfg['batch_size'], shuffle=False)

# Load model
base_model = BERT4Rec(
    num_user=len(domain_data),
    num_item=num_item,
    hidden_units=base_cfg['hidden_units'],
    num_heads=base_cfg['num_heads'],
    num_layers=base_cfg['num_layers'],
    max_len=base_cfg['max_len'],
    dropout_rate=base_cfg['dropout_rate'],
    device=base_cfg['device'],
)
model = DANN_BERT4Rec(base_model).to(base_cfg['device'])
model.load_state_dict(torch.load(infer_cfg['checkpoint_path'], map_location=base_cfg['device']))
model.eval()

# Extract pooled embeddings
embeddings_specific = []
embeddings_common = []

with torch.no_grad():
    for seq in data_loader:
        seq = seq.to(base_cfg['device'])
        emb_s, emb_c = model.extract_embeddings(seq)  # ⬅️ 올바르게 pooled 벡터 획득
        embeddings_specific.append(emb_s.cpu().numpy())
        embeddings_common.append(emb_c.cpu().numpy())

emb_s = np.vstack(embeddings_specific)
emb_c = np.vstack(embeddings_common)

# Save
os.makedirs('embeddings', exist_ok=True)
pickle.dump(emb_s, open(data_cfg['target_spe_emb'], 'wb'))
pickle.dump(emb_c, open(data_cfg['target_cross_emb'], 'wb'))
