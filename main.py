import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


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


def calculate_correlations_and_plot(data):
    data_2022 = data[data["entrées 2022"] > 0]

    correlation_screens = data_2022["écrans"].corr(data_2022["entrées 2022"])
    correlation_seats = data_2022["fauteuils"].corr(data_2022["entrées 2022"])

    print(
        "Corrélation entre le nombre d'écrans et les entrées annuelles (2022):",
        correlation_screens,
    )
    print(
        "Corrélation entre le nombre de fauteuils et les entrées annuelles (2022):",
        correlation_seats,
    )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.regplot(
        x="écrans", y="entrées 2022", data=data_2022, scatter_kws={"alpha": 0.6}
    )
    plt.title("Corrélation: Écrans vs Entrées annuelles (2022)")

    plt.subplot(1, 2, 2)
    sns.regplot(
        x="fauteuils", y="entrées 2022", data=data_2022, scatter_kws={"alpha": 0.6}
    )
    plt.title("Corrélation: Fauteuils vs Entrées annuelles (2022)")

    plt.tight_layout()
    plt.show()


def prepare_data_for_modeling(data):
    filtered_data = data.loc[
        :,
        [
            "écrans",
            "fauteuils",
            "population de la commune",
            "entrées 2021",
        ],
    ]

    filtered_data = filtered_data.dropna()

    X = filtered_data[["écrans", "fauteuils", "population de la commune"]]
    y = filtered_data[["entrées 2021"]].mean(axis=1)

    return X, y


def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print(f"R² sur l'ensemble d'entraînement : {train_r2:.2f}")
    print(f"R² sur l'ensemble de test : {test_r2:.2f}")
    print(f"Erreur moyenne absolue (MAE) sur l'ensemble de test : {test_mae:.2f}")

    return model


def test_model_on_2022_data(model, data):
    X_2022 = data[["écrans", "fauteuils", "population de la commune"]]
    y_2022 = data["entrées 2022"]

    y_pred_2022 = model.predict(X_2022)

    comparison = pd.DataFrame(
        {"Valeurs réelles (2022)": y_2022, "Prédictions": y_pred_2022}
    )

    print("Comparaison des prédictions avec les valeurs réelles (2022) :")
    print(comparison.head(10))


cinema_data = get_clean_cinema_data("data/cinemas.csv")

print("Aperçu des données nettoyées :")
print(cinema_data.head())

# print("\nStatistiques descriptives :")
# print(cinema_data[["fauteuils", "écrans", "entrées 2021", "entrées 2022"]].describe())

# region_statistics = calculate_region_statistics(cinema_data)

# display_top_and_bottom_regions(region_statistics)

# plot_top_regions(region_statistics)

# calculate_correlations_and_plot(cinema_data)

X, y = prepare_data_for_modeling(cinema_data)

print("Entraînement et évaluation du modèle :")
model = train_and_evaluate_model(X, y)

print("Test du modèle sur les données de 2022 :")
test_model_on_2022_data(model, cinema_data)
