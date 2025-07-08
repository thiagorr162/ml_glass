import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
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
save_path = pathlib.Path("viz/iforest")
save_path.mkdir(parents=True, exist_ok=True)

# Carregando os dados
df = pd.read_csv(data_path)

# Dividindo em treino e teste
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Separando X e y
train_X = train_df.drop(columns=["RefractiveIndex"])
test_X = test_df.drop(columns=["RefractiveIndex"])
y_test = test_df["RefractiveIndex"]

# Treinando Isolation Forest
iso_forest = IsolationForest(contamination=0.05, n_estimators=800, random_state=42)
iso_forest.fit(train_X)

# Predizendo outliers no conjunto de teste
outlier_pred = iso_forest.predict(test_X)
outlier_score = iso_forest.decision_function(test_X)

# Adiciona colunas de outlier e score ao dataframe
test_X = test_X.copy()
test_X["outlier"] = outlier_pred  # -1 para anômalo, 1 para normal
test_X["outlier_score"] = outlier_score

# Aplicando t-SNE
print("Aplicando t-SNE aos dados do teste...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embedding = tsne.fit_transform(test_X.drop(columns=["outlier", "outlier_score"]))

# Plot
plt.figure(figsize=(8, 6))
for val, marker, color in [(-1, "x", "black"), (1, "o", "tab:green")]:
    idx = test_X["outlier"] == val
    plt.scatter(
        embedding[idx, 0],
        embedding[idx, 1],
        label=f"{'Anômalo' if val == -1 else 'Normal'}",
        marker=marker,
        color=color,
        s=40,
        edgecolor="k",
        alpha=1.0 if val == -1 else 0.2,
    )

plt.title("t-SNE dos Dados de Teste com Pontos Anômalos via Isolation Forest", fontsize=14)
plt.xlabel("Componente t-SNE 1", fontsize=12)
plt.ylabel("Componente t-SNE 2", fontsize=12)
plt.legend(title="Classe", fontsize=11, title_fontsize=12)
plt.tight_layout()
plt.savefig(save_path / "tsne_iforest_outliers_yshape.png", dpi=300)
plt.show()
