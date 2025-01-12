import pandas as pd
import matplotlib.pyplot as plt


def get_clean_cinema_data(cinema_path):
    return (
        pd.read_csv(cinema_path, sep=";", encoding="utf-8", engine="python")
        .drop_duplicates()
        .fillna(
            {
                "population de la commune": 0,
                "écrans": 0,
                "fauteuils": 0,
                "entrées 2021": 0,
                "entrées 2022": 0,
                "label Art et Essai": "non",
            }
        )
        .astype(
            {
                "population de la commune": int,
                "écrans": int,
                "fauteuils": int,
                "entrées 2021": int,
                "entrées 2022": int,
            }
        )
        .assign(
            label_art_et_essai=lambda x: x["label Art et Essai"].str.strip().str.lower()
        )
    )


def calculate_region_statistics(data):
    return (
        data.groupby("région administrative")
        .agg(
            total_entrees_2022=("entrées 2022", "sum"),
            total_fauteuils=("fauteuils", "sum"),
        )
        .assign(
            avg_entrees_per_fauteuil=lambda x: x["total_entrees_2022"]
            / x["total_fauteuils"]
        )
        .sort_values(by="avg_entrees_per_fauteuil", ascending=False)
    )


def display_top_and_bottom_regions(statistics, top_n=3):
    top_regions = statistics.head(top_n)
    bottom_regions = statistics.tail(top_n)

    print("3 meilleures régions en termes d'entrées moyennes par fauteuil :")
    print(top_regions)

    print("\n3 pires régions en termes d'entrées moyennes par fauteuil :")
    print(bottom_regions)


def plot_top_regions(statistics, top_n=10):
    top_regions = statistics.head(top_n)
    plt.figure(figsize=(10, 6))
    plt.bar(
        top_regions.index,
        top_regions["avg_entrees_per_fauteuil"],
        color="skyblue",
    )
    plt.xlabel("Région administrative")
    plt.ylabel("Entrées moyennes par fauteuil")
    plt.title(f"Top {top_n} des régions : Entrées moyennes par fauteuil (2022)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


cinema_data = get_clean_cinema_data("data/cinemas.csv")

print("Aperçu des données nettoyées :")
print(cinema_data.head())

print("\nStatistiques descriptives :")
print(cinema_data[["fauteuils", "écrans", "entrées 2021", "entrées 2022"]].describe())

region_statistics = calculate_region_statistics(cinema_data)

display_top_and_bottom_regions(region_statistics)

plot_top_regions(region_statistics)
