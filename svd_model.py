# -*- coding: utf-8 -*-
"""
SVD-based recommender system - Lokalna verzija
Originalni Colab: https://colab.research.google.com/drive/1IstoFX8JoRo61YzOPMlF4AhK60Cr7RCF
"""

import pandas as pd
import numpy as np
import pickle
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ================= KONFIGURACIJA =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Putanje do podataka
TRAINING_FILE = os.path.join(DATA_DIR, 'dataset1', 'archive')
FILE = os.path.join(DATA_DIR, 'dataset1', 'archive', 'probe.txt')
MODEL_FILENAME = os.path.join(MODEL_SAVE_DIR, "svd_model5.pt")

# ================= FUNKCIJE =================
def load_training_data_userwise(nrows=None, random_state=42):
    ratings = []
    for part in range(1, 5):  # combined_data_1.txt
        filename = os.path.join(TRAINING_FILE, f"combined_data_{part}.txt")
        with open(filename, "r") as f:
            movie_id = None
            for line in f:
                if nrows and len(ratings) >= nrows:
                    break
                line = line.strip()
                if line.endswith(':'):
                    movie_id = int(line[:-1])
                else:
                    user_id, rating, _ = line.split(",")
                    ratings.append([int(user_id), movie_id, float(rating)])
    
    df = pd.DataFrame(ratings, columns=["userID", "itemID", "rating"])

    # Podjela na train/val/test
    train_list, val_list, test_list = [], [], []
    grouped = df.groupby("userID")
    for user_id, group in grouped:
        group = group.sample(frac=1, random_state=random_state)
        n = len(group)
        train_end = int(n * 0.7)
        val_end = train_end + int(n * 0.2)
        
        train_list.append(group.iloc[:train_end])
        val_list.append(group.iloc[train_end:val_end])
        test_list.append(group.iloc[val_end:])
    
    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    
    test_df.to_csv(os.path.join(DATA_DIR, "forTest_svd5.csv"), index=False)
    return train_df, val_df, test_df

def evaluate_probe(model, eval_df, verbose=True):
    y_true = eval_df["rating"].tolist()
    y_pred = [model.predict(row["userID"], row["itemID"]).est for _, row in eval_df.iterrows()]
    
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    
    if verbose:
        print(f"ℹ️ Evaluirano {len(eval_df)} ocjena")
        print(f"📊 RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    return rmse, mae

# ================= GLAVNI KOD =================
if __name__ == "__main__":
    print("📥 Učitavanje podataka...")
    train_df, val_df, test_df = load_training_data_userwise(nrows=55_000_000)  
    
    print("\n📊 Statistički pregled:")
    print(f"Trening skup: {len(train_df)} recenzija")
    print(f"Validacioni skup: {len(val_df)} recenzija")
    print(f"Test skup: {len(test_df)} recenzija")
    print(f"Korisnika: {train_df['userID'].nunique()}")
    print(f"Filmova: {train_df['itemID'].nunique()}\n")
    
    # Priprema podataka za Surprise
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_df[['userID', 'itemID', 'rating']], reader)
    trainset = data.build_full_trainset()
    
    # Treniranje modela
    print("🎯 Treniranje SVD modela...")
    model = SVD(
        n_factors=100,
        n_epochs=20,  # Smanjeno za brže izvršavanje
        lr_all=0.005,
        reg_all=0.02,
        verbose=True
    )
    model.fit(trainset)
    
    # Evaluacija
    print("\n🔍 Evaluacija modela:")
    rmse_val, mae_val = evaluate_probe(model, val_df)
    
    # Čuvanje modela
    with open(MODEL_FILENAME, "wb") as f:
        pickle.dump(model, f)
    print(f"\n💾 Model sačuvan u: {MODEL_FILENAME}")