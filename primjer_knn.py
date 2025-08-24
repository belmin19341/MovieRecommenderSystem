# -*- coding: utf-8 -*-

import os
import pandas as pd

# --- Pomoćna funkcija za učitavanje movies_titles.csv ---
# Ako nazivi filmova sadrže zareze, ovo rješava problem
def load_movies_with_commas(filepath):
    records = []
    with open(filepath, "r", encoding="ISO-8859-1") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                itemID = int(parts[0])
                year = parts[1]
                title = ",".join(parts[2:])
                records.append((itemID, year, title))
    return pd.DataFrame(records, columns=["itemID", "year", "title"])


# --- Glavna funkcija za pregled preporuka jednog korisnika ---
def pregled_preporuka_jednog_korisnika(test_csv_path, movies_csv_path, user_id, k=10):
    df = pd.read_csv(test_csv_path)
    movies_df = load_movies_with_commas(movies_csv_path)

    user_df = df[df["userID"] == user_id]
    if user_df.empty:
        print(f"Nema podataka za korisnika s ID-em: {user_id}")
        return

    top_k = user_df.sort_values("predicted_rating", ascending=False).head(k)
    merged = top_k.merge(movies_df, on="itemID", how="left")

    print(f"\n🎥 Preporuke za korisnika {user_id} (Top-{k}):\n")
    for _, row in merged.iterrows():
        print(
            f"🎬 {row['title']} | ID: {row['itemID']} | "
            f"✅ Stvarna ocjena: {row['rating']} | 🤖 Predikcija: {row['predicted_rating']:.2f}"
        )


# --- Funkcija za prikaz filmova koje STVARNO korisnik najviše voli ---
def pregled_omiljenih_filmova_korisnika(test_csv_path, movies_csv_path, user_id, k=10):
    df = pd.read_csv(test_csv_path)
    movies_df = load_movies_with_commas(movies_csv_path)

    user_df = df[df["userID"] == user_id]
    if user_df.empty:
        print(f"Nema stvarnih ocjena za korisnika s ID-em: {user_id}")
        return

    top_k = user_df.sort_values("rating", ascending=False).head(k)
    merged = top_k.merge(movies_df, on="itemID", how="left")

    print(f"\n❤️ Omiljeni filmovi korisnika {user_id} (Top-{k} po stvarnim ocjenama):\n")
    for _, row in merged.iterrows():
        print(
            f"🎬 {row['title']} | ID: {row['itemID']} | "
            f"⭐ Ocjena: {row['rating']} | 🤖 Predikcija: {row['predicted_rating']:.2f}"
        )


if __name__ == "__main__":
    # Podesi lokalne putanje
    putanja = os.path.abspath(".")  # trenutni folder
    test_csv = os.path.join(putanja, "forTest_with_predictions.csv")
    movies_csv = os.path.join(putanja, "data/dataset1/archive/movie_titles.csv")

    # Pregled preporuka za jednog korisnika
    pregled_preporuka_jednog_korisnika(
        test_csv_path=test_csv,
        movies_csv_path=movies_csv,
        user_id=6,
        k=5
    )

    # Pregled stvarnih najdražih filmova istog korisnika
    pregled_omiljenih_filmova_korisnika(
        test_csv_path=test_csv,
        movies_csv_path=movies_csv,
        user_id=6,
        k=5
    )

    # Pregled prvih 75 filmova (ako želiš debugirati CSV)
    movies_df = pd.read_csv(
        movies_csv,
        sep="\t",  # Ako datoteka nije tab-delimited, promijeni u ","
        header=None,
        encoding="ISO-8859-1",
        engine="python"
    )

    print(movies_df.head(75))
    print(movies_df.shape)
