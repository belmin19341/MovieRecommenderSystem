# -*- coding: utf-8 -*-

import os
import pickle
import pandas as pd
from surprise import SVD
from sklearn.metrics import mean_squared_error, mean_absolute_error, ndcg_score

# Putanje
putanja = os.path.abspath(".")  # trenutni folder
MODEL_SVD_SAVE_DIR = os.path.join(putanja, "saved_models")
MODEL_PATH = os.path.join(MODEL_SVD_SAVE_DIR, "knn_model10.pt")
TEST_CSV_PATH = os.path.join(putanja,"data", "forTest_knn10.csv")

# Učitaj test podatke
test_df = pd.read_csv(TEST_CSV_PATH)

# Učitaj spremljeni model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Generiši predikcije
predictions = []
for _, row in test_df.iterrows():
    uid, iid = int(row["userID"]), int(row["itemID"])
    pred = model.predict(uid, iid)
    predictions.append(pred.est)

# Dodaj predikcije u DataFrame i sačuvaj
test_df["predicted_rating"] = predictions
test_df.to_csv("forTest_with_predictions10.csv", index=False)

# Izračunaj RMSE i MAE
rmse = mean_squared_error(test_df["rating"], test_df["predicted_rating"], squared=True)
mae = mean_absolute_error(test_df["rating"], test_df["predicted_rating"])

print("📊 Test skup metrike:")
print(f"  • RMSE: {rmse:.4f}")
print(f"  • MAE: {mae:.4f}")

# Funkcija za NDCG@k
def calculate_ndcg_at_k(csv_path, k=5):
    df = pd.read_csv(csv_path)
    ndcg_values = []

    grouped = df.groupby("userID")
    for _, group in grouped:
        if len(group) < k:
            continue

        top_k = group.sort_values("predicted_rating", ascending=False).head(k)
        true_relevance = top_k["rating"].values.reshape(1, -1)
        predicted_scores = top_k["predicted_rating"].values.reshape(1, -1)

        ndcg = ndcg_score(true_relevance, predicted_scores, k=k)
        ndcg_values.append(ndcg)

    return sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0

ndcg_at_5 = calculate_ndcg_at_k("forTest_with_predictions10.csv", k=5)
print(f"🎯 NDCG@5: {ndcg_at_5:.4f}")
