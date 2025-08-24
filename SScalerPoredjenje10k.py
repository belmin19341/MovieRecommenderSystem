import pandas as pd
import pickle
from surprise import Dataset, Reader, KNNBasic
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

CSV_FILE = os.path.join(DATA_DIR, 'myDatasetStandardized.csv')

def load_custom_csv(use_std='none', nrows=None, random_state=42):
    """use_std: 'none' | 'global' | 'user'"""
    df = pd.read_csv(CSV_FILE, nrows=nrows,
                     names=['userID', 'itemID', 'rating', 'rating_std_global', 'rating_std_user'],
                     header=None)

    # Konvertuj sve rating kolone u numeričke vrijednosti
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating_std_global'] = pd.to_numeric(df['rating_std_global'], errors='coerce')
    df['rating_std_user'] = pd.to_numeric(df['rating_std_user'], errors='coerce')
    
    # Ukloni NaN vrijednosti ako postoje
    df = df.dropna(subset=['rating', 'rating_std_global', 'rating_std_user'])
    
    if use_std == 'global':
        df = df[['userID', 'itemID', 'rating_std_global']]
        df = df.rename(columns={'rating_std_global':'rating'})
    elif use_std == 'user':
        df = df[['userID', 'itemID', 'rating_std_user']]
        df = df.rename(columns={'rating_std_user':'rating'})
    else:
        df = df[['userID', 'itemID', 'rating']]
    
    # Podjela train/val/test kao ranije
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
    
    # Dodaj provjeru tipova podataka
    print(f"Tip podataka u rating koloni: {train_df['rating'].dtype}")
    print(f"Min rating: {train_df['rating'].min()}, Max rating: {train_df['rating'].max()}")
    
    return train_df, val_df, test_df

def evaluate_model(train_df, val_df, k=40, min_k=5, model_name="knn"):
    # Provjeri da li su rating vrijednosti numeričke
    if not pd.api.types.is_numeric_dtype(train_df['rating']):
        train_df['rating'] = pd.to_numeric(train_df['rating'], errors='coerce')
        val_df['rating'] = pd.to_numeric(val_df['rating'], errors='coerce')
        
        # Ukloni NaN vrijednosti
        train_df = train_df.dropna(subset=['rating'])
        val_df = val_df.dropna(subset=['rating'])
    
    # Odredi rating scale na osnovu podataka
    min_rating = train_df['rating'].min()
    max_rating = train_df['rating'].max()
    
    print(f"Rating range: {min_rating} to {max_rating}")
    
    if min_rating < 0:
        reader = Reader(rating_scale=(min_rating, max_rating))
    else:
        reader = Reader(rating_scale=(min_rating, max_rating))
    
    data = Dataset.load_from_df(train_df[['userID', 'itemID', 'rating']], reader)
    trainset = data.build_full_trainset()

    sim_options = {"name": "cosine", "user_based": False}
    model = KNNBasic(k=k, min_k=min_k, sim_options=sim_options, verbose=True)
    model.fit(trainset)
    
    # Evaluacija na val
    y_true = val_df["rating"].tolist()
    y_pred = [model.predict(row["userID"], row["itemID"]).est for _, row in val_df.iterrows()]
    
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"📊 {model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return rmse, mae, model

if __name__ == "__main__":
    # 1. Bez standardizacije
    print("\n=== Originalne ocjene ===")
    train_df, val_df, test_df = load_custom_csv(use_std='none', nrows=100000)
    evaluate_model(train_df, val_df, model_name="KNN-original")

    # 2. Globalna standardizacija
    print("\n=== Globalno standardizovane ocjene ===")
    train_df, val_df, test_df = load_custom_csv(use_std='global', nrows=100000)
    evaluate_model(train_df, val_df, model_name="KNN-global-std")

    # 3. Per-user standardizacija
    print("\n=== Per-user standardizovane ocjene ===")
    train_df, val_df, test_df = load_custom_csv(use_std='user', nrows=100000)
    evaluate_model(train_df, val_df, model_name="KNN-user-std")