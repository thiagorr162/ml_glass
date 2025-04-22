import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Argumentos de linha de comando
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
viz_path = pathlib.Path("viz/conformal")
viz_path.mkdir(parents=True, exist_ok=True)

# Carregando os dados
df = pd.read_csv(data_path)
X = df.drop(columns=["RefractiveIndex"])
y = np.minimum(df["RefractiveIndex"].values, 4)  # Truncando o y em 4

# Separando os dados em treino, calibração e teste
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_calib, y_train, y_calib = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

# Treinamento do modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Previsões no conjunto de calibração
y_calib_pred = model.predict(X_calib)
nonconformity_scores = np.abs(y_calib - y_calib_pred)

# Quantil ajustado
alpha = 0.01
n_calib = len(nonconformity_scores)
quantile_index = int(np.ceil((1 - alpha) * (n_calib + 1))) - 1
quantile_index = min(max(quantile_index, 0), n_calib - 1)
threshold = np.sort(nonconformity_scores)[quantile_index]

# Previsões no conjunto de teste
y_test_pred = model.predict(X_test)

# Construção dos intervalos conformes
lower_bounds = y_test_pred - threshold
upper_bounds = y_test_pred + threshold

# Gerando figura com intervalos preditivos
plt.figure(figsize=(8, 6))
order = np.argsort(y_test)
y_sorted = y_test[order]
y_pred_sorted = y_test_pred[order]
lower_sorted = lower_bounds[order]
upper_sorted = upper_bounds[order]

plt.plot(y_sorted, y_pred_sorted, "o", color="tab:blue", label="Previsão")
plt.fill_between(y_sorted, lower_sorted, upper_sorted, color="lightblue", alpha=0.5, label="Intervalo conforme")
plt.plot([0, 4], [0, 4], linestyle="--", color="gray", label="$x = y$")
plt.xlabel("Valor real", fontsize=12)
plt.ylabel("Valor predito", fontsize=12)
plt.title("Intervalos preditivos via conformal prediction", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(viz_path / "conformal_prediction_intervals.png", dpi=300)
plt.close()
