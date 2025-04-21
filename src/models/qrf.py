import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.model_selection import train_test_split

# Parser para o nome do arquivo
parser = argparse.ArgumentParser()
parser.add_argument(
    "--file_name",
    default="processed_ND300-RefractiveIndex.csv",
    help="Data file name.",
    type=str,
)
args = parser.parse_args()

# Caminhos
data_path = pathlib.Path(f"data/processed/{args.file_name}")
save_path = pathlib.Path("viz/qrf")
save_path.mkdir(parents=True, exist_ok=True)

# Carregando os dados
df = pd.read_csv(data_path)

# Separando features (X) e target (y)
X = df.drop(columns=["RefractiveIndex"])
y = df["RefractiveIndex"].copy()

# Truncagem de y: limitando valores > 4 para 4
y = np.minimum(y, 4)

# Dividindo os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo
model = RandomForestQuantileRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Previsões
quantiles = [0.05, 0.5, 0.95]
predictions = model.predict(X_test, quantiles=quantiles)

# Organizando os dados
results_df = pd.DataFrame(
    {
        "y_true": y_test.values,
        "y_pred_low": predictions[:, 0],
        "y_pred_med": predictions[:, 1],
        "y_pred_upp": predictions[:, 2],
    }
)

# Ordenando pelos valores reais
results_df_sorted = results_df.sort_values(by="y_true").reset_index(drop=True)

# Criando o gráfico
plt.figure(figsize=(8, 6))

# Faixa de confiança (entre os quantis 0.05 e 0.95)
plt.fill_between(
    results_df_sorted["y_true"],
    results_df_sorted["y_pred_low"],
    results_df_sorted["y_pred_upp"],
    color="gray",
    alpha=0.3,
    label="Intervalo 90%",
)

# Previsão (mediana)
plt.scatter(
    results_df_sorted["y_true"], results_df_sorted["y_pred_med"], color="royalblue", s=30, label="Previsão (mediana)"
)

# Reta y = x (previsão ideal)
min_val = min(results_df_sorted["y_true"].min(), results_df_sorted["y_pred_med"].min())
max_val = max(results_df_sorted["y_true"].max(), results_df_sorted["y_pred_med"].max())
plt.plot([min_val, max_val], [min_val, max_val], "k--", label="Previsão ideal (y = x)")

# Eixos e título
plt.xlabel("Valor real (Y)", fontsize=16)
plt.ylabel("Previsão (mediana)", fontsize=16)
plt.title("Previsão vs Valor Real com Intervalos de Confiança", fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()

# Salvando o gráfico
plt.savefig(save_path / "qrf_intervals.png", dpi=300)
plt.close()
