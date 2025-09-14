import matplotlib.pyplot as plt

def plot_top10_recommendations(test_csv_path, movies_csv_path, user_id, model_name, out_path):
    df = pd.read_csv(test_csv_path)
    movies_df = load_movies_with_commas(movies_csv_path)
    user_df = df[df["userID"] == user_id]

    if user_df.empty:
        print(f"Nema podataka za korisnika {user_id}")
        return

    # Top-10 predikcija
    top_pred = user_df.sort_values("predicted_rating", ascending=False).head(10)
    merged_pred = top_pred.merge(movies_df, on="itemID", how="left")

    # Top-10 stvarnih ocjena
    top_true = user_df.sort_values("rating", ascending=False).head(10)
    merged_true = top_true.merge(movies_df, on="itemID", how="left")

    # Računaj hit-rate (koliko filmova iz predikcija je i u top stvarnih)
    hit_count = len(set(merged_pred["itemID"]) & set(merged_true["itemID"]))
    hit_rate = hit_count / 10 * 100

    # Napravi plot
    plt.figure(figsize=(8,6))
    plt.barh(merged_pred["title"], merged_pred["predicted_rating"], color="skyblue")
    plt.gca().invert_yaxis()
    plt.title(f"Top-10 preporuka za korisnika {user_id} ({model_name})\nHit-rate: {hit_count}/10 ({hit_rate:.0f}%)")
    plt.xlabel("Predikcija ocjene")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
