import os
import pandas as pd
import matplotlib.pyplot as plt

# ================= KONFIGURACIJA =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAINING_FILE = os.path.join(DATA_DIR, 'dataset1', 'archive')

# ================= FUNKCIJE =================
def load_data_flat(nrows=None):
    ratings = []
    for part in range(1, 5):  # kao u heatmap skripti
        filename = os.path.join(TRAINING_FILE, f"combined_data_{part}.txt")
        if not os.path.exists(filename):
            print(f"⚠️ Fajl {filename} ne postoji, preskačem...")
            continue
            
        with open(filename, "r") as f:
            movie_id = None
            for line in f:
                if nrows and len(ratings) >= nrows:
                    break
                line = line.strip()
                if line.endswith(':'):
                    movie_id = int(line[:-1])
                else:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        user_id, rating = parts[0], parts[1]
                        ratings.append([int(user_id), movie_id, float(rating)])
    
    if not ratings:
        raise ValueError("Nema podataka za učitavanje!")
    
    return pd.DataFrame(ratings, columns=["userID", "itemID", "rating"])

# ================= GLAVNI KOD =================
if __name__ == "__main__":
    print("📥 Učitavanje podataka (sample 100.000 ocjena)...")
    df = load_data_flat(nrows=100_000)
    print(f"Učitano: {len(df):,} ocjena")
    print(f"Korisnika: {df['userID'].nunique():,}")
    print(f"Filmova: {df['itemID'].nunique():,}")

    # Izračun gustine matrice
    density = len(df) / (df['userID'].nunique() * df['itemID'].nunique())
    print(f"Gustina matrice: {density:.6f} ({density*100:.4f}%)")

    # Scatter plot korisnik-film
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df["userID"], 
        df["itemID"], 
        s=5, alpha=0.5, edgecolor='none'
    )
    plt.title(
        f"Vizualizacija sparsnosti matrice "
        f"(uzorak: {df['userID'].nunique()} korisnika, "
        f"{df['itemID'].nunique()} filmova, {len(df)} ocjena)\n"
        f"Gustina: {density:.6f} ({density*100:.4f}%)"
    )
    plt.xlabel("Korisnik ID")
    plt.ylabel("Film ID")
    plt.tight_layout()

    # Spasi graf
    out_path = os.path.join(DATA_DIR, "sparsity_plot.png")
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"✅ Scatter plot sačuvan u: {out_path}")
