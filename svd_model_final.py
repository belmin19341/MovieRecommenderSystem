# -*- coding: utf-8 -*-
"""
SVD-based recommender system - memory-optimized + Precision/Recall
Radi u koracima da laptop ne zaglavi.
"""

import pandas as pd
import numpy as np
import pickle
from surprise import Dataset, Reader, SVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# ================= KONFIGURACIJA =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_FILE = os.path.join(DATA_DIR, 'myDatasetStandardized.csv')
TRAIN_SPLIT = os.path.join(DATA_DIR, "train_svd_standardized.csv")
VAL_SPLIT = os.path.join(DATA_DIR, "val_svd_standardized.csv")
TEST_SPLIT = os.path.join(DATA_DIR, "forTest_svd_standardized.csv")
MODEL_FILENAME = os.path.join(MODEL_SAVE_DIR, "svd_model_standardized.pt")


# ================= FUNKCIJE =================
def load_or_create_split(random_state=42):
    """Ako već postoje splitani CSV-ovi, učitaj ih; inače napravi i spremi."""
    if os.path.exists(TRAIN_SPLIT) and os.path.exists(VAL_SPLIT) and os.path.exists(TEST_SPLIT):
        print("📂 Učitavam postojeće splitove...")
        train_df = pd.read_csv(TRAIN_SPLIT)
        val_df = pd.read_csv(VAL_SPLIT)
        test_df = pd.read_csv(TEST_SPLIT)
        return train_df, val_df, test_df

    print("📥 Kreiram split po korisniku (70/20/10)...")
    df = pd.read_csv(CSV_FILE, nrows=50_000_000)
    train_list, val_list, test_list = [], [], []
    grouped = df.groupby("userID")
    for user_id, group in grouped:
        group = group.sample(frac=1, random_state=random_state)
        n = len(group)
        if n < 3:
            train_list.append(group)
            continue
        train_end = int(n * 0.7)
        val_end = train_end + int(n * 0.2)
        train_list.append(group.iloc[:train_end])
        val_list.append(group.iloc[train_end:val_end])
        test_list.append(group.iloc[val_end:])

    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    train_df.to_csv(TRAIN_SPLIT, index=False)
    val_df.to_csv(VAL_SPLIT, index=False)
    test_df.to_csv(TEST_SPLIT, index=False)

    print("✅ Split sačuvan.")
    return train_df, val_df, test_df


def precision_recall_chunked(model, eval_df, k=10, threshold=3.5, chunk_size=200_000):
    """Računa RMSE, MAE, Precision@k i Recall@k u blokovima da štedi RAM."""
    y_true_all, y_pred_all = [], []
    user_est_true = defaultdict(list)
    n = len(eval_df)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = eval_df.iloc[start:end]
        for _, row in chunk.iterrows():
            pred = model.predict(row["userID"], row["itemID"], row["rating"])
            y_true_all.append(row["rating"])
            y_pred_all.append(pred.est)
            user_est_true[pred.uid].append((pred.est, row["rating"]))
        print(f"  ...obrađen chunk {start}-{end}/{n}")

    # RMSE i MAE
    rmse = mean_squared_error(y_true_all, y_pred_all, squared=False)
    mae = mean_absolute_error(y_true_all, y_pred_all)

    # Precision i Recall
    precisions, recalls = {}, {}
    for uid, ratings in user_est_true.items():
        ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum(true_r >= threshold for (_, true_r) in ratings)
        n_rec_k = sum(est >= threshold for (est, _) in ratings[:k])
        n_rel_and_rec_k = sum((true_r >= threshold and est >= threshold)
                              for (est, true_r) in ratings[:k])
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel else 0

    precision = sum(precisions.values()) / len(precisions) if precisions else 0
    recall = sum(recalls.values()) / len(recalls) if recalls else 0
    return rmse, mae, precision, recall


# ================= GLAVNI KOD =================
if __name__ == "__main__":
    # 1) Split ili load CSV
    train_df, val_df, test_df = load_or_create_split()
    print(f"Trening skup: {len(train_df)}, Validacioni skup: {len(val_df)}, Test skup: {len(test_df)}")

    # 2) Treniraj SVD
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_df[['userID', 'itemID', 'rating']], reader)
    trainset = data.build_full_trainset()

    print("\n🎯 Treniranje SVD modela...")
    model = SVD(
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        verbose=True
    )
    model.fit(trainset)

    # 3) Evaluacija na trening skupu (RMSE i MAE samo – prec/rec obično nisu bitni ovdje)
    print("\n📊 Evaluacija na trening skupu:")
    train_rmse, train_mae, _, _ = precision_recall_chunked(model, train_df, k=10)

    # 4) Evaluacija na validation skupu
    print("\n🔍 Evaluacija na validacionom skupu:")
    val_rmse, val_mae, prec_val, rec_val = precision_recall_chunked(model, val_df, k=10)
    print(f"RMSE validacija: {val_rmse:.4f}, MAE: {val_mae:.4f}")
    print(f"Precision@10: {prec_val:.4f}, Recall@10: {rec_val:.4f}")

    # 5) Sačuvaj model
    with open(MODEL_FILENAME, "wb") as f:
        pickle.dump(model, f)
    print(f"\n💾 Model sačuvan u: {MODEL_FILENAME}")

    # 6) Graf – uporedni prikaz trening vs validacionih grešaka
    metrics = ['RMSE', 'MAE']
    train_scores = [train_rmse, train_mae]
    val_scores = [val_rmse, val_mae]

    plt.figure(figsize=(7, 5))
    x = np.arange(len(metrics))
    plt.bar(x - 0.15, train_scores, width=0.3, label='Train', alpha=0.7)
    plt.bar(x + 0.15, val_scores, width=0.3, label='Validation', alpha=0.7)
    plt.xticks(x, metrics)
    plt.ylabel("Vrijednost greške")
    plt.title("Uporedni prikaz trening vs validacionih grešaka (SVD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "svd_train_vs_val.png"), dpi=300)
    plt.close()
    print(f"📊 Graf sačuvan u: {os.path.join(RESULTS_DIR, 'svd_train_vs_val.png')}")
