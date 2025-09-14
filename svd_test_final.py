# -*- coding: utf-8 -*-

import os
import pickle
import pandas as pd
import numpy as np
from surprise import SVD
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score
from collections import defaultdict

# ================= KONFIGURACIJA =================
BASE_DIR = os.path.abspath(".")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "svd_model_standardized.pt")
TEST_CSV_PATH = os.path.join(DATA_DIR, "forTest_svd_standardized.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "forTest_with_predictions_svd_standardized1.csv")


# ================= FUNKCIJE =================
def precision_recall_at_k(test_df, k=10, threshold=3.5):
    """
    Računa Precision@k i Recall@k za već generisane predikcije.
    """
    precisions, recalls = {}, {}

    grouped = test_df.groupby("userID")
    for uid, group in grouped:
        # Sortiraj po predikciji
        group_sorted = group.sort_values("predicted_rating", ascending=False)
        top_k = group_sorted.head(k)

        n_rel = sum(group["rating"] >= threshold)
        n_rec_k = sum(top_k["predicted_rating"] >= threshold)
        n_rel_and_rec_k = sum((top_k["rating"] >= threshold) & (top_k["predicted_rating"] >= threshold))

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel else 0

    precision = sum(precisions.values()) / len(precisions) if precisions else 0
    recall = sum(recalls.values()) / len(recalls) if recalls else 0
    return precision, recall


def calculate_ndcg_at_k(test_df, k=10):
    """
    Računa prosječni NDCG@k za sve korisnike.
    """
    ndcg_values = []
    grouped = test_df.groupby("userID")
    for _, group in grouped:
        if len(group) < k:
            continue

        top_k = group.sort_values("predicted_rating", ascending=False).head(k)
        true_relevance = top_k["rating"].values.reshape(1, -1)
        predicted_scores = top_k["predicted_rating"].values.reshape(1, -1)

        ndcg = ndcg_score(true_relevance, predicted_scores, k=k)
        ndcg_values.append(ndcg)

    return sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0


# ================= GLAVNI KOD =================
if __name__ == "__main__":
    # 1) Učitaj test podatke
    test_df = pd.read_csv(TEST_CSV_PATH)

    # 2) Učitaj spremljeni model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # 3) Generiši predikcije
    predictions = []
    for _, row in test_df.iterrows():
        uid, iid = int(row["userID"]), int(row["itemID"])
        pred = model.predict(uid, iid)
        predictions.append(pred.est)

    test_df["predicted_rating"] = predictions
    test_df.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 Sačuvan test CSV sa predikcijama: {OUTPUT_CSV}")

    # 4) Izračunaj osnovne metrike
    rmse = mean_squared_error(test_df["rating"], test_df["predicted_rating"], squared=False)
    mae = mean_absolute_error(test_df["rating"], test_df["predicted_rating"])

    # 5) Precision@k i Recall@k
    precision, recall = precision_recall_at_k(test_df, k=10)

    # 6) NDCG@10
    ndcg_at_10 = calculate_ndcg_at_k(test_df, k=10)

    # 7) Prikaži rezultate
    print("\n📊 Test skup metrike:")
    print(f"  • RMSE: {rmse:.4f}")
    print(f"  • MAE: {mae:.4f}")
    print(f"  • Precision@10: {precision:.4f}")
    print(f"  • Recall@10: {recall:.4f}")
    print(f"  • NDCG@10: {ndcg_at_10:.4f}")
