# -*- coding: utf-8 -*-
"""
Vizualizacija sparsnosti matrice ocjena - poboljšana verzija
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================= KONFIGURACIJA =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAINING_FILE = os.path.join(DATA_DIR, 'dataset1', 'archive')

# ================= FUNKCIJE =================
def load_training_data_userwise(nrows=None, random_state=42):
    ratings = []
    for part in range(1, 5):
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
    print("📥 Učitavanje podataka...")
    df = load_training_data_userwise(nrows=100_000)
    
    print(f"Učitano: {len(df):,} ocjena")
    print(f"Korisnika: {df['userID'].nunique():,}")
    print(f"Filmova: {df['itemID'].nunique()}")
    
    # Izračun gustine matrice
    density = len(df) / (df['userID'].nunique() * df['itemID'].nunique())
    print(f"Gustina matrice: {density:.6f} ({density*100:.4f}%)")
    
    # 1. HEATMAP VIZUALIZACIJA
    plt.figure(figsize=(12, 8))
    
    # Pripremi podatke za heatmap
    pivot_data = df.pivot_table(index='userID', columns='itemID', 
                               values='rating', aggfunc='count', 
                               fill_value=0)
    
    # Kreiraj heatmap
    plt.imshow(pivot_data.values, aspect='auto', cmap='Reds')
    plt.colorbar(label='Broj ocjena')
    plt.title(f"Heatmap rijetkosti matrice\nGustina: {density*100:.4f}%")
    plt.xlabel("Film ID")
    plt.ylabel("Korisnik ID")
    
    # Označi filmove na x-osi
    plt.xticks(range(len(pivot_data.columns)), pivot_data.columns, rotation=45)
    plt.tight_layout()
    
    # Spasi heatmap
    heatmap_path = os.path.join(DATA_DIR, "sparsity_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap sačuvan u: {heatmap_path}")
    
    # 2. DISTRIBUCUA OCJENA PO FILMOVIMA
    plt.figure(figsize=(10, 6))
    
    item_rating_counts = df['itemID'].value_counts().sort_index()
    plt.bar(range(len(item_rating_counts)), item_rating_counts.values)
    plt.title('Broj ocjena po filmu')
    plt.xlabel('Film ID')
    plt.ylabel('Broj ocjena')
    plt.xticks(range(len(item_rating_counts)), item_rating_counts.index)
    
    # Dodaj brojeve iznad stupaca
    for i, v in enumerate(item_rating_counts.values):
        plt.text(i, v + max(item_rating_counts.values)*0.01, str(v), 
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Spasi bar plot
    barplot_path = os.path.join(DATA_DIR, "ratings_per_item.png")
    plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Bar plot sačuvan u: {barplot_path}")
    
    plt.show()
    
    # Dodatna statistika
    print("\n📊 Detaljna statistika:")
    print(f"Ukupno korisnika: {df['userID'].nunique():,}")
    print(f"Ukupno filmova: {df['itemID'].nunique()}")
    print(f"Ukupno ocjena: {len(df):,}")
    print(f"Gustina matrice: {density*100:.6f}%")
    print(f"Prosječno ocjena po korisniku: {len(df)/df['userID'].nunique():.2f}")
    print(f"Prosječno ocjena po filmu: {len(df)/df['itemID'].nunique():.2f}")
    
    # Distribucija ocjena po filmu
    print("\n📊 Ocjene po filmu:")
    for item_id, count in df['itemID'].value_counts().sort_index().items():
        print(f"Film {item_id}: {count} ocjena ({count/len(df)*100:.1f}%)")