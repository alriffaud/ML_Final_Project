#!/usr/bin/env python3
"""
This script loads a SAKT model and runs inference on a user.
It loads the model architecture and weights from the specified paths, and the
QID-to-index mapping from a pickle file.
It then processes the user's interaction data, updating the history and mask as
new interactions are encountered.
Finally, it predicts the probability of the next correct answer using the SAKT
model, and provides general reinforcement based on the model's
predictions and the user's performance.
"""
import json
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import random
import numpy as np
import pandas as pd
from collections import defaultdict
build_sakt = __import__("build").build_sakt


# —————— Constants ——————
MIN_HISTORY = 10
P_LOW, P_MED = 0.5, 0.8
MASTERY_THRESHOLD = 0.7
P_LOW_CUM = 0.5

# —————— Clean data loading ——————
# Paths
path = os.path.join("..", "Data_Preprocessing", "data")
inter_df = pd.read_parquet(os.path.join(path, "interactions_clean.parquet"))
quest_df = pd.read_parquet(os.path.join(path, "questions_clean.parquet"))

# —————— SAKT Loading + qid2idx ——————
# Paths
base = os.path.join("..", "Knowledge_Tracing_Models", "4_SAKT", "export")
sakt_architecture_path = os.path.join(base, "sakt_architecture.json")
sakt_weights_path = os.path.join(base, "sakt_weights.h5")
sakt_meta_path = os.path.join(base, "sakt_meta.json")
qid2idx_path = os.path.join(base, "qid2idx.pkl")
# Load meta
meta = json.load(open(sakt_meta_path, "r"))
E = meta["E"]
max_len = meta["max_len"]
d_model = meta["d_model"]
dropout = meta["dropout"]
# batch_size does not matter for building the model

# We rebuild the identical architecture
model = build_sakt(E=E, d_model=d_model, max_len=max_len, dropout=dropout)
# We load the weights
model.load_weights(sakt_weights_path)
# We load the mapping
with open(qid2idx_path, "rb") as f:
    qid2idx = pickle.load(f)

# —————— Mapping load QID→CATEGORY ——————
cat_map = dict(zip(quest_df["id"], quest_df["general_cat"]))


# Auxiliary function for coding a response
def update_history(hist, mask, qid, correct):
    """
    Update the history and mask with a new interaction.
    Args:
        hist: list of integers (qid + E * correct)
        mask: list of floats (1.0 for each interaction)
        qid: integer, question ID
        correct: boolean, whether the answer was correct
    Returns:
        hist: updated list of integers
        mask: updated list of floats
    """
    token = qid2idx[qid] + (int(correct) * E)
    if len(hist) < max_len:
        hist.append(token)
        mask.append(1.0)
    else:
        # we scroll to the left and put at the end
        hist.pop(0)
        mask.pop(0)
        hist.append(token)
        mask.append(1.0)
    return hist, mask


def predict_next(hist, mask):
    """
    Construct arrays X,M of shape (1, max_len) with padding on the left,
    and returns p_next = model.predict(...)
    Args:
        hist: list of integers (qid + E * correct)
        mask: list of floats (1.0 for each interaction)
    Returns:
        p_next: float, probability of the next correct answer
    """
    # burn-in: with <MIN_HISTORY interactions we return 0.5
    if len(hist) < MIN_HISTORY:
        return 0.5
    X = np.zeros((1, max_len), dtype=np.int32)
    M = np.zeros((1, max_len), dtype=np.float32)
    L = len(hist)
    if L > 0:
        X[0, -L:] = hist
        M[0, -L:] = mask
    # prediction
    return model.predict([X, M], verbose=0)[0, 0]


# We choose a user at random
uid = random.choice(inter_df["user_id"].unique())
user_df = inter_df[inter_df["user_id"] == uid].sort_values("start_time")

print(f"\n=== Report for user {uid} ({len(user_df)} total interactions) ===\n")

# We initialize the history and mask
hist, mask = [], []

cum_total = 0
cum_correct = 0

# We group by quiz_id and process each quiz session.
for quiz_id, grp in user_df.groupby("quiz_id"):
    if grp.empty:
        continue
    # Número de interacciones en este quiz
    n_q = len(grp)
    # Número de aciertos en este quiz
    correct_q = grp['correct'].sum()    # True→1, False→0

    # Actualizo los acumulados
    cum_total += n_q
    cum_correct += correct_q

    # Calculo porcentaje acumulado
    pct_cum = cum_correct / cum_total

    for _, row in grp.iterrows():
        qid = row["question_id"]
        corr = bool(row["correct"])
        hist, mask = update_history(hist, mask, qid, corr)

    # SAKT global prediction
    p_global = predict_next(hist, mask)

    # mastery by category in this quiz
    counts = defaultdict(lambda: {"correct": 0, "total": 0})
    for _, row in grp.iterrows():
        qid = row["question_id"]
        sc = row["score"]
        cat = cat_map.get(qid, "unknown")
        counts[cat]["total"] += 1
        if sc > 0:
            counts[cat]["correct"] += 1
    mastery = {c: counts[c]["correct"]/counts[c]["total"]
               for c in counts}

    # prints results
    print(f"Quiz {quiz_id}:  P(next correct) = {p_global:.1%}")
    print(f"🎯  Cumulative accuracy: {pct_cum:.1%} \
({int(cum_correct)}/{cum_total})")
    print("📊  Performance by category:")
    for cat, pct in mastery.items():
        corr = counts[cat]["correct"]
        tot = counts[cat]["total"]
        print(f"    - {cat:15s}: {pct:.0%} ({corr}/{tot})")
    # weak categories
    weak_cats = [c for c, p in mastery.items() if p < MASTERY_THRESHOLD]
    print("🧠 → Topics to review:", ", ".join(weak_cats) or "None")

    # if SAKT predicts low confidence, give general reinforcement
    if p_global < P_LOW or pct_cum < P_LOW_CUM:
        print("⚠️  Your overall readiness is low — consider \
revisiting fundamentals.\n")
    elif p_global < P_MED and len(weak_cats) > 0:
        print("👍  Your overall readiness is good, but consider revisiting \
the topics listed above.\n")
    else:
        print("✅  You seem ready for the next challenge!\n")
    print()
