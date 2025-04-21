import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

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
save_path = pathlib.Path("viz/qrf")
save_path.mkdir(parents=True, exist_ok=True)

# Carregando os dados
df = pd.read_csv(data_path)

# Separando features (X) e target (y)
X = df.copy()
y = np.minimum(df["RefractiveIndex"].values, 4)

# Definindo 2 classes com base no índice de refração
classes = []
for val in y:
    if val <= 1.8:
        classes.append("Y ≤ 1.8")
    else:
        classes.append("Y > 1.8")
classes = np.array(classes)

# t-SNE aplicado aos dados originais completos
print("Aplicando t-SNE aos dados originais (todos)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embedding = tsne.fit_transform(X)

# Plot
plt.figure(figsize=(8, 6))
colors = ["tab:blue", "tab:red"]
for label, color in zip(np.unique(classes), colors):
    idx = classes == label
    plt.scatter(embedding[idx, 0], embedding[idx, 1], label=label, color=color, s=40, edgecolor="k", alpha=0.8)

plt.title("t-SNE dos Dados Originais (Todos)\nClasses baseadas no Índice de Refração", fontsize=14)
plt.xlabel("Componente t-SNE 1", fontsize=12)
plt.ylabel("Componente t-SNE 2", fontsize=12)
plt.legend(title="Classe", fontsize=11, title_fontsize=12)
plt.tight_layout()

# Salvar
plt.savefig(save_path / "tsne_dados_originais_2classes.png", dpi=300)
plt.show()
