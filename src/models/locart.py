import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Treinamento do modelo base
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Previsões para calibração
y_calib_pred = model.predict(X_calib)
residuals = np.abs(y_calib - y_calib_pred)

# Treinamento da árvore nos resíduos
partition_tree = DecisionTreeRegressor(max_leaf_nodes=10, random_state=42)
partition_tree.fit(X_calib, residuals)

# Identificando partições na calibração
partitions = partition_tree.apply(X_calib)
partition_thresholds = {}
alpha = 0.01

for node in np.unique(partitions):
    node_residuals = residuals[partitions == node]
    n = len(node_residuals)
    if n == 0:
        continue
    q_idx = int(np.ceil((1 - alpha) * (n + 1))) - 1
    q_idx = min(max(q_idx, 0), n - 1)
    threshold = np.sort(node_residuals)[q_idx]
    partition_thresholds[node] = threshold

# Previsões no conjunto de teste
y_test_pred = model.predict(X_test)
test_partitions = partition_tree.apply(X_test)

lower_bounds = []
upper_bounds = []

for pred, node in zip(y_test_pred, test_partitions):
    threshold = partition_thresholds.get(node, 0.0)
    lower_bounds.append(pred - threshold)
    upper_bounds.append(pred + threshold)

lower_bounds = np.array(lower_bounds)
upper_bounds = np.array(upper_bounds)

# Gerando figura com intervalos preditivos adaptativos
plt.figure(figsize=(8, 6))
order = np.argsort(y_test)
y_sorted = y_test[order]
y_pred_sorted = y_test_pred[order]
lower_sorted = lower_bounds[order]
upper_sorted = upper_bounds[order]

plt.plot(y_sorted, y_pred_sorted, "o", color="tab:blue", label="Previsão")
plt.fill_between(y_sorted, lower_sorted, upper_sorted, color="lightblue", alpha=0.5, label="Intervalo LOCART")
plt.plot([0, 4], [0, 4], linestyle="--", color="gray", label="$x = y$")
plt.xlabel("Valor real", fontsize=12)
plt.ylabel("Valor predito", fontsize=12)
plt.title("Intervalos adaptativos via LOCART", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(viz_path / "locart_prediction_intervals.png", dpi=300)
plt.close()
