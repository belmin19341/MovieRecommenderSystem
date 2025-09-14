import pandas as pd

# --- parametri ---
input_file = "data/forTest_with_predictions_svd_standardized1.csv"  # CSV fajl sa kolona: userId, movieId, rating, prediction
output_file = "model_stats_users_svd.csv"
target_users = [1664539]

print("📥 Učitavam podatke...")
df = pd.read_csv(input_file)

results = []

for user_id in target_users:
    user_data = df[df['userID'] == user_id]
    n_ratings = len(user_data)
    if n_ratings == 0:
        print(f"⚠ Korisnik {user_id} nema ocjena u ovom setu!")
        continue

    # sortiraj stvarne ocjene (opadajuće)
    user_sorted_real = user_data.sort_values(by='rating', ascending=False)
    top_half_count = n_ratings // 2 if n_ratings > 1 else 1
    top_half_movies = user_sorted_real.head(top_half_count)['itemID'].tolist()

    # sortiraj po predikcijama (opadajuće)
    user_sorted_pred = user_data.sort_values(by='predicted_rating', ascending=False)
    top_predictions = user_sorted_pred.head(100)  # uzmi top-100 predikcija

    # koliko iz važnije polovine je u top-100 predikcija
    hits = top_predictions[top_predictions['itemID'].isin(top_half_movies)]
    hit_count = len(hits)
    hit_percent = hit_count / top_half_count * 100

    # koliko je model pogodio tačno mjesto (indeks unutar te polovine)
    # rang u stvarnom setu
    real_rank = {mid: rank for rank, mid in enumerate(user_sorted_real.head(top_half_count)['itemID'].tolist(), start=1)}
    pred_rank = {mid: rank for rank, mid in enumerate(top_predictions['itemID'].tolist(), start=1)}

    correct_position_count = 0
    for mid in top_half_movies:
        if mid in pred_rank and mid in real_rank and pred_rank[mid] == real_rank[mid]:
            correct_position_count += 1

    correct_position_percent = correct_position_count / top_half_count * 100

    results.append({
        "userId": user_id,
        "total_ratings": n_ratings,
        "top_half_count": top_half_count,
        "hit_count": hit_count,
        "hit_percent": round(hit_percent, 2),
        "correct_position_count": correct_position_count,
        "correct_position_percent": round(correct_position_percent, 2)
    })

# napravi tabelu i snimi CSV
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False)
print(f"✅ Rezultati su snimljeni u '{output_file}'")
print(results_df)
