import pandas as pd


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


cinema_data = get_clean_cinema_data("data/cinemas.csv")

print("Aperçu des données nettoyées :")
print(cinema_data.head())

print("\nStatistiques descriptives :")
print(cinema_data[["fauteuils", "écrans", "entrées 2021", "entrées 2022"]].describe())
