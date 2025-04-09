
# ADR-Rec: Adaptive Disentanglement for Cross-Domain Sequential Recommendation

This project implements a cross-domain sequential recommendation system combining **BERT4Rec**, **Domain-Adversarial Neural Networks (DANN)**, and **Cross-Attention Gating** mechanisms.

---

## 📁 Project Structure

```
ADR_Rec/
├── config.yaml                  # Central configuration file
├── utils/
│   └── config_loader.py         # Utility to load YAML config
├── src/
│   ├── model/
│   │   ├── bert4rec.py          # BERT4Rec model
│   │   ├── dann.py              # DANN model + embedding extraction
│   │   └── gatedrec.py          # Cross-attention recommendation model
│   └── dataset/
│       └── datasets.py          # Dataset & preprocessing
├── infer/
│   └── embeddings.py            # Embedding inference using DANN
├── rec_model/
│   └── train_eval.py            # Final training & evaluation script
├── train.py                     # Train the DANN model
```

---

## ⚙️ Configuration (`config.yaml`)

```yaml
base:
  hidden_units: 50
  num_heads: 2
  num_layers: 2
  max_len: 64
  dropout_rate: 0.1
  device: "cuda"

train:
  lr: 0.001
  batch_size: 256
  num_epochs: 100
  mask_prob: 0.2
  alpha: 0.5

recommend:
  lr: 0.0005
  batch_size: 256
  num_epochs: 20
  checkpoint_output: "checkpoints/gatedrec.pt"

infer:
  batch_size: 128
  checkpoint_path: "checkpoints/B4RDANN_epoch100.pt"

data:
  source_path: "data/source.pkl"
  target_path: "data/target.pkl"
  cross_path: "data/cross.pkl"
  target_spe_emb: "embeddings/spe_emb.pkl"
  target_cross_emb: "embeddings/cross_emb.pkl"
```

---

## 🧪 How to Run

### 1. Train DANN model
```bash
PYTHONPATH=. python train.py
```

### 2. Extract embeddings
```bash
PYTHONPATH=. python infer/embeddings.py
```

### 3. Train & Evaluate Recommendation Model
```bash
PYTHONPATH=. python rec_model/train_eval.py
```

---

## 🧠 Core Concepts

- **BERT4Rec**: Sequence-aware item recommendation model
- **DANN**: Disentangles domain-specific vs domain-invariant user representations
- **Cross-Attention Gating**: Fuses both embeddings via attention-based gating mechanism

---

## ⚠️ Notes

- CrossEntropyLoss requires labels to be in range `[0, num_item]`
- All parameters & paths are loaded from `config.yaml`
- `evaluate()` applies index clipping to avoid CUDA errors

---

## 📄 License
For research use only. Contact us for commercial licensing.

---

## 📂 Data Description

This project operates on two domains — for example, **Food** and **Kitchen**. The goal is to build a model that recommends items in the **target domain** (e.g., Food) using auxiliary data from a **source domain** (e.g., Kitchen).

### 🔗 Data File Setup Example

If we are recommending for the **Food** domain using **Kitchen** as the source, config paths are set as:

```yaml
data:
  target_path: data/food&kitchen/food_train.pkl        # Food domain training data (validation item excluded)
  source_path: data/food&kitchen/kitchen_all.pkl       # Kitchen domain full data
  cross_path: data/food&kitchen/cross_f_train.pkl       # Cross-domain training embeddings (f → food)
```

- The `f` in `cross_f_train.pkl` stands for **food** (target).
- `target_path` must exclude the last item per user for validation.
- Similarly, the corresponding entries in `cross_path` must align with the truncated `target_path` data.

This ensures that embedding extraction and recommendation are consistent with the training-validation split.

