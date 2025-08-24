# -*- coding: utf-8 -*-
"""
KNN – poređenje originalnih i standardizovanih ocjena (user/global)
Učitava `data/myDatasetStandardized.csv` [userID,itemID,rating,rating_std_global,rating_std_user]
radi istu podjelu (70/20/10 po korisniku), trenira KNNBasic (item-based),
evaluira na val/test i snima rezultate + modele.

Dodane Precision@k i Recall@k metrike.
"""

import os
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from surprise import Dataset, Reader, KNNBasic
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ================= KONFIGURACIJA =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_SAVE_DIR = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

CSV_FILE = os.path.join(DATA_DIR, 'myDatasetStandardized.csv')
RESULTS_FILE = os.path.join(DATA_DIR, 'knn_std_comparison.csv')

# ======= POMOĆNE FUNKCIJE ZA PRECISION I RECALL =======
def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Izračunaj Precision@k i Recall@k za Surprise predictions.
    
    Args:
        predictions: Lista Surprise Prediction objekata
        k: Broj preporuka za svakog korisnika
        threshold: Prag za pozitivnu ocjenu
    
    Returns:
        Tuple (precision@k, recall@k) kao float vrijednosti
    """
    # Mapa za praćenje top-k predviđenih i relevantnih itema po korisniku
    user_est_true = defaultdict(list)
    
    # Grupiraj predviđanja po korisniku
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    
    precisions = dict()
    recalls = dict()
    
    for uid, user_ratings in user_est_true.items():
        # Sortiraj predviđene ocjene silazno
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        # Broj relevantnih itema za ovog korisnika
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        
        # Broj preporučenih itema u top-k koji su relevantni
        n_rec_k = sum((est >= threshold) for (est, true_r) in user_ratings[:k])
        
        # Broj preporučenih itema u top-k
        n_rec_k_total = min(len(user_ratings), k)
        
        # Precision@k: proporcija preporučenih itema koji su relevantni
        precisions[uid] = n_rec_k / n_rec_k_total if n_rec_k_total > 0 else 0
        
        # Recall@k: proporcija relevantnih itema koji su preporučeni
        recalls[uid] = n_rec_k / n_rel if n_rel > 0 else 0
    
    # Prosječni precision i recall preko svih korisnika
    avg_precision = sum(prec for prec in precisions.values()) / len(precisions) if precisions else 0
    avg_recall = sum(rec for rec in recalls.values()) / len(recalls) if recalls else 0
    
    return avg_precision, avg_recall

def get_top_n(predictions, n=10, threshold=3.5):
    """Vrati top-N preporuke za svakog korisnika."""
    top_n = defaultdict(list)
    
    for uid, iid, true_r, est, _ in predictions:
        if est >= threshold:
            top_n[uid].append((iid, est))
    
    # Sortiraj preporuke po procijenjenoj ocjeni i uzmi top-N
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n

def calculate_coverage(top_n, all_items):
    """Izračunaj pokrivenost (coverage) - proporcija itema koji su preporučeni."""
    recommended_items = set()
    for uid, items in top_n.items():
        for iid, _ in items:
            recommended_items.add(iid)
    
    return len(recommended_items) / len(all_items) if all_items else 0

# ======= POMOĆNE =======
def split_userwise(df, random_state=42):
    """70/20/10 po korisniku, isti princip kao u tvom KNN kodu."""
    train_list, val_list, test_list = [], [], []
    for _, g in df.groupby("userID"):
        g = g.sample(frac=1.0, random_state=random_state)
        n = len(g)
        i1 = int(n * 0.7)
        i2 = i1 + int(n * 0.2)
        train_list.append(g.iloc[:i1])
        val_list.append(g.iloc[i1:i2])
        test_list.append(g.iloc[i2:])
    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df   = pd.concat(val_list).reset_index(drop=True)
    test_df  = pd.concat(test_list).reset_index(drop=True)
    return train_df, val_df, test_df

def build_reader_from_range(series):
    """Surprise zahtijeva rating_scale=(min,max). Napravi ga iz stvarnog raspona."""
    rmin = float(series.min())
    rmax = float(series.max())
    if rmin == rmax:
        rmin -= 1.0
        rmax += 1.0
    return Reader(rating_scale=(rmin, rmax))

def train_eval_knn(train_df, val_df, test_df, model_tag, k=40, min_k=5, eval_k=10, threshold=3.5):
    """Treniraj KNNBasic (item-based) i evaluiraj na val/test. Vrati metrike i model."""
    reader = build_reader_from_range(train_df['rating'])
    data = Dataset.load_from_df(train_df[['userID', 'itemID', 'rating']], reader)
    trainset = data.build_full_trainset()

    sim_options = {"name": "cosine", "user_based": False}
    model = KNNBasic(k=k, min_k=min_k, sim_options=sim_options, verbose=True)
    model.fit(trainset)

    # Eval funkcija za RMSE i MAE
    def _eval(split_df):
        y_true = split_df["rating"].tolist()
        y_pred = [model.predict(row["userID"], row["itemID"]).est for _, row in split_df.iterrows()]
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        return rmse, mae

    rmse_val, mae_val = _eval(val_df)
    rmse_test, mae_test = _eval(test_df)

    # Eval funkcija za Precision@k i Recall@k
    def _eval_ranking(split_df, k=10, threshold=3.5):
        # Napravi predictions za cijeli split
        predictions = []
        for _, row in split_df.iterrows():
            pred = model.predict(row["userID"], row["itemID"])
            predictions.append((pred.uid, pred.iid, pred.r_ui, pred.est, {}))
        
        precision, recall = precision_recall_at_k(predictions, k=k, threshold=threshold)
        
        # Izračunaj pokrivenost
        all_items = set(train_df['itemID'].unique())
        top_n = get_top_n(predictions, n=k, threshold=threshold)
        coverage = calculate_coverage(top_n, all_items)
        
        return precision, recall, coverage

    # Evaluacija ranking metrika na test skupu
    precision_test, recall_test, coverage_test = _eval_ranking(test_df, k=eval_k, threshold=threshold)

    # Snimi model
    model_path = os.path.join(MODEL_SAVE_DIR, f"knn_{model_tag}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ {model_tag}:")
    print(f"   VAL  - RMSE={rmse_val:.4f} MAE={mae_val:.4f}")
    print(f"   TEST - RMSE={rmse_test:.4f} MAE={mae_test:.4f}")
    print(f"   RANKING (k={eval_k}) - Precision={precision_test:.4f} Recall={recall_test:.4f} Coverage={coverage_test:.4f}")

    return {
        "model": model_tag,
        "train_min": float(train_df['rating'].min()),
        "train_max": float(train_df['rating'].max()),
        "val_rmse": rmse_val,
        "val_mae": mae_val,
        "test_rmse": rmse_test,
        "test_mae": mae_test,
        f"precision@{eval_k}": precision_test,
        f"recall@{eval_k}": recall_test,
        f"coverage@{eval_k}": coverage_test,
        "model_path": model_path,
    }

# ================= GLAVNI TOK =================
if __name__ == "__main__":
    # Konfiguracija parametara
    K_VALUE = 40
    MIN_K = 5
    EVAL_K = 10  # Broj preporuka za Precision@k i Recall@k
    THRESHOLD = 3.5  # Prag za pozitivnu ocjenu

    print(f"📥 Učitavam {CSV_FILE} ...")
    df_full = pd.read_csv(
        CSV_FILE,
        header=None,
        names=["userID", "itemID", "rating", "rating_std_global", "rating_std_user"],
        nrows=50_000_000
    )
    print(f"➡️  Učitano: {len(df_full)} redova | korisnika={df_full.userID.nunique()} | filmova={df_full.itemID.nunique()}")

    # --- 1) ORIGINAL ---
    print(f"\n=== ORIGINAL (rating 1..5) - k={K_VALUE}, min_k={MIN_K} ===")
    df_orig = df_full[["userID", "itemID", "rating"]].copy()
    tr_o, va_o, te_o = split_userwise(df_orig)
    res_orig = train_eval_knn(tr_o, va_o, te_o, model_tag="original", 
                             k=K_VALUE, min_k=MIN_K, eval_k=EVAL_K, threshold=THRESHOLD)

    # --- 2) GLOBAL STD ---
    print(f"\n=== GLOBAL STD - k={K_VALUE}, min_k={MIN_K} ===")
    df_g = df_full[["userID", "itemID", "rating_std_global"]].rename(columns={"rating_std_global": "rating"})
    tr_g, va_g, te_g = split_userwise(df_g)
    res_g = train_eval_knn(tr_g, va_g, te_g, model_tag="std_global", 
                          k=K_VALUE, min_k=MIN_K, eval_k=EVAL_K, threshold=THRESHOLD)

    # --- 3) USER STD ---
    print(f"\n=== USER STD - k={K_VALUE}, min_k={MIN_K} ===")
    df_u = df_full[["userID", "itemID", "rating_std_user"]].rename(columns={"rating_std_user": "rating"})
    tr_u, va_u, te_u = split_userwise(df_u)
    res_u = train_eval_knn(tr_u, va_u, te_u, model_tag="std_user", 
                          k=K_VALUE, min_k=MIN_K, eval_k=EVAL_K, threshold=THRESHOLD)

    # Snimi rezultate u CSV
    results_df = pd.DataFrame([res_orig, res_g, res_u])
    results_df["train_size"] = [len(tr_o), len(tr_g), len(tr_u)]
    results_df["val_size"] = [len(va_o), len(va_g), len(va_u)]
    results_df["test_size"] = [len(te_o), len(te_g), len(te_u)]
    results_df.to_csv(RESULTS_FILE, index=False)

    print(f"\n📄 Rezultati upisani u: {RESULTS_FILE}")
    print("🧠 Modeli snimljeni u:", MODEL_SAVE_DIR)
    
    # Prikaz rezultata
    print("\n" + "="*80)
    print("REZULTATI EVALUACIJE:")
    print("="*80)
    for _, row in results_df.iterrows():
        print(f"\n{row['model']}:")
        print(f"  RMSE (test): {row['test_rmse']:.4f}")
        print(f"  MAE (test): {row['test_mae']:.4f}")
        print(f"  Precision@{EVAL_K}: {row[f'precision@{EVAL_K}']:.4f}")
        print(f"  Recall@{EVAL_K}: {row[f'recall@{EVAL_K}']:.4f}")
        print(f"  Coverage@{EVAL_K}: {row[f'coverage@{EVAL_K}']:.4f}")