"""
Training pipeline for the word classifier.

Loads session data, builds IKT + L-R feature vectors,
trains the MLP, and evaluates top-1 and top-5 accuracy.

Usage:
    PYTHONPATH=. python src/model/train.py \
        --session data/raw/word_sessions/session_day1.json
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

from src.analysis.vocabulary_ambiguity import get_lr_pattern, LEFT_KEYS, RIGHT_KEYS
from src.model.classifier import WordClassifier


# ── constants ─────────────────────────────────────────────────────────────────
MAX_LEN   = 12    # pad/truncate IKT sequences to this length
N_AUGMENT = 8     # extra augmented copies per real sample
SEED      = 42


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ── feature engineering ───────────────────────────────────────────────────────

def lr_one_hot(word: str, max_len: int) -> list:
    """
    Binary L-R vector: 1 = right hand, 0 = left hand.
    Padded to max_len.
    """
    vec = []
    for ch in word.lower():
        if ch in RIGHT_KEYS:
            vec.append(1.0)
        elif ch in LEFT_KEYS:
            vec.append(0.0)
    # pad
    vec = vec[:max_len]
    vec += [0.0] * (max_len - len(vec))
    return vec


def normalise_ikts(ikts: list, max_len: int) -> list:
    """Min-max normalise IKTs and pad to max_len."""
    if not ikts:
        return [0.0] * max_len
    arr = np.array(ikts, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    arr = arr[:max_len]
    pad = np.zeros(max_len - len(arr))
    return np.concatenate([arr, pad]).tolist()


def make_feature(ikts: list, word: str, max_len: int) -> list:
    """Concatenate normalised IKTs + L-R one-hot."""
    return normalise_ikts(ikts, max_len) + lr_one_hot(word, max_len)


def augment(ikts: list, n: int, noise_std: float = 0.05) -> list:
    """
    Generate `n` augmented copies by adding Gaussian noise.
    Simulates natural variation in typing rhythm.
    """
    arr = np.array(ikts, dtype=float)
    copies = []
    for _ in range(n):
        noise = np.random.normal(0, noise_std * (arr.mean() + 1e-6), size=arr.shape)
        augmented = np.clip(arr + noise, 1.0, None).tolist()
        copies.append(augmented)
    return copies


# ── data loading ──────────────────────────────────────────────────────────────

def load_dataset(session_path: str, max_len: int, augment_n: int):
    with open(session_path) as f:
        session = json.load(f)

    X, y, words_list = [], [], []

    for word, entry in session["words"].items():
        reps = [r["ikts"] for r in entry["repetitions"] if r["ikts"]]
        if not reps:
            continue

        for ikts in reps:
            X.append(make_feature(ikts, word, max_len))
            y.append(word)

        # augment from real samples
        for ikts in reps:
            for aug in augment(ikts, augment_n):
                X.append(make_feature(aug, word, max_len))
                y.append(word)

        words_list.append(word)

    return np.array(X, dtype=np.float32), y, words_list


# ── train / eval ──────────────────────────────────────────────────────────────

def train_and_evaluate(session_path: str, epochs: int = 150,
                       test_frac: float = 0.2):
    set_seed(SEED)

    print(f"\n  Loading data from {session_path} ...")
    X_raw, y_raw, vocabulary = load_dataset(session_path, MAX_LEN, N_AUGMENT)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    input_dim   = X_raw.shape[1]

    print(f"  Words: {num_classes}   Samples: {len(X_raw)}"
          f"   Input dim: {input_dim}")

    # Train/test split (stratified by word)
    indices = list(range(len(X_raw)))
    random.shuffle(indices)
    split   = int(len(indices) * (1 - test_frac))
    tr_idx, te_idx = indices[:split], indices[split:]

    X_tr = torch.tensor(X_raw[tr_idx])
    y_tr = torch.tensor(y_enc[tr_idx], dtype=torch.long)
    X_te = torch.tensor(X_raw[te_idx])
    y_te = torch.tensor(y_enc[te_idx], dtype=torch.long)

    loader = DataLoader(TensorDataset(X_tr, y_tr),
                        batch_size=32, shuffle=True)

    model   = WordClassifier(input_dim, num_classes)
    optim   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    sched   = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.5)

    # Training loop
    print(f"\n  Training for {epochs} epochs ...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        sched.step()

        if epoch % 30 == 0 or epoch == 1:
            avg = total_loss / len(loader)
            print(f"  epoch {epoch:3d}  loss {avg:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X_te)

    # Top-1 accuracy
    top1 = (logits.argmax(dim=1) == y_te).float().mean().item()

    # Top-5 accuracy
    top5_preds = logits.topk(min(5, num_classes), dim=1).indices
    top5 = sum(
        y_te[i].item() in top5_preds[i].tolist()
        for i in range(len(y_te))
    ) / len(y_te)

    print(f"\n  ── Results ──────────────────────────────")
    print(f"  Test samples : {len(y_te)}")
    print(f"  Top-1 acc    : {top1:.1%}")
    print(f"  Top-5 acc    : {top5:.1%}")
    print(f"  Baseline     : {1/num_classes:.1%}  (random)")
    print(f"  ─────────────────────────────────────────")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state":  model.state_dict(),
        "label_encoder": list(le.classes_),
        "input_dim":    input_dim,
        "num_classes":  num_classes,
        "top1_acc":     top1,
        "top5_acc":     top5,
    }, "models/classifier.pt")
    print(f"  Model saved → models/classifier.pt")

    return top1, top5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--epochs",  type=int, default=150)
    args = parser.parse_args()
    train_and_evaluate(args.session, args.epochs)


if __name__ == "__main__":
    main()