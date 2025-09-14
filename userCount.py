import pandas as pd

# --- ulazni fajlovi ---
input_files = {
    "svd": "data/forTest_with_predictions_svd_standardized1.csv",
    "knn": "data/forTest_with_predictions_knn_standardized1.csv"
}
output_file = "userCount.csv"

all_results = []

for model_name, input_file in input_files.items():
    print(f"📥 Učitavam podatke za model {model_name.upper()} iz '{input_file}'...")
    df = pd.read_csv(input_file)

    # broj ocjena po korisniku
    user_counts = df.groupby("userID").size().reset_index(name="rating_count")

    # top-10 najviše i najmanje
    top_users = user_counts.sort_values(by="rating_count", ascending=False).head(10)
    bottom_users = user_counts.sort_values(by="rating_count", ascending=True).head(10)

    # spoji u jedan dataframe sa oznakom tipa i modela
    top_users["type"] = "top_10"
    bottom_users["type"] = "bottom_10"
    top_users["model"] = model_name
    bottom_users["model"] = model_name

    all_results.append(pd.concat([top_users, bottom_users], ignore_index=True))

# spoji rezultate oba modela u jedan dataframe
final_df = pd.concat(all_results, ignore_index=True)

# snimi u CSV
final_df.to_csv(output_file, index=False)
print(f"✅ Rezultati spremljeni u '{output_file}'")
print(final_df)
