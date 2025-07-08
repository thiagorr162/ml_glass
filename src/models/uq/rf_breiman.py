import argparse
import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
save_path = pathlib.Path("csv/breiman")
save_path.mkdir(parents=True, exist_ok=True)

# Carregando os dados
df = pd.read_csv(data_path)

# Separando features (X) e target (y)
X = df.drop(columns=["RefractiveIndex"])
y = np.minimum(df["RefractiveIndex"].values, 4)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Treinando a RandomForest
rf = RandomForestRegressor(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)

# Calculando folhas
leaf_train = rf.apply(X_train)  # (n_train, n_trees)
leaf_test = rf.apply(X_test)  # (n_test, n_trees)

n_train = leaf_train.shape[0]
n_test = leaf_test.shape[0]
n_trees = leaf_train.shape[1]

# Calculando matriz de similaridade de Breiman entre teste e treino
similarity_matrix = np.zeros((n_test, n_train))
print("Calculando similaridade de Breiman...")
for t in tqdm(range(n_trees)):
    train_leaf = leaf_train[:, t]
    test_leaf = leaf_test[:, t]
    leaf_to_train = defaultdict(list)
    for i, leaf in enumerate(train_leaf):
        leaf_to_train[leaf].append(i)
    for j, leaf in enumerate(test_leaf):
        for i in leaf_to_train.get(leaf, []):
            similarity_matrix[j, i] += 1
similarity_matrix /= n_trees

# Seleciona os 10 pontos de teste com maior y
top_10_idx = np.argsort(-y_test)[:10]

# Para cada um, encontra os 3 pontos de treino mais similares e gera uma tabela
for idx in top_10_idx:
    similares = np.argsort(-similarity_matrix[idx])[:3]
    vidro_teste = X_test.iloc[idx]
    comp_teste = vidro_teste[vidro_teste > 0]
    y_teste = y_test[idx]

    dados = []
    for i in similares:
        vidro_similar = X_train.iloc[i]
        comp_similar = vidro_similar[vidro_similar > 0]
        y_similar = y_train[i]
        sim_breiman = similarity_matrix[idx, i]

        dados.append(
            {
                "vidro_teste_comp": ", ".join(f"{k}: {v:.2f}" for k, v in comp_teste.items()),
                "y_teste": y_teste,
                "vidro_similar_comp": ", ".join(f"{k}: {v:.2f}" for k, v in comp_similar.items()),
                "y_similar": y_similar,
                "similaridade_breiman": f"{sim_breiman:.3f}",
            }
        )

    tabela = pd.DataFrame(dados)
    tabela.to_csv(save_path / f"vidro_{idx}_top3_similares.csv", index=False)
    print(f"Salvo: vidro_{idx}_top3_similares.csv")
