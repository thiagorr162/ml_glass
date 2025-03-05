import argparse
import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
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

# Carregando os dados
data_path = pathlib.Path(f"data/processed/{args.file_name}")
df = pd.read_csv(data_path)

# Separando features (X) e target (y)
X = df.drop(columns=["RefractiveIndex"])
y = df["RefractiveIndex"].copy()

# Truncagem de y: limitando valores > 4 para 4
y = np.minimum(y, 4)

# Dividindo os dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o RandomForestQuantileRegressor
model = RandomForestQuantileRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Fazendo previsões para o conjunto de teste
quantiles = [0.05, 0.5, 0.95]  # Inferior, mediana e superior
predictions = model.predict(X_test, quantiles=quantiles)

# Extraindo os valores preditos
y_pred_lower = predictions[:, 0]  # Quantil 5%
y_pred_median = predictions[:, 1]  # Mediana (50%)
y_pred_upper = predictions[:, 2]  # Quantil 95%

# Criando um DataFrame para visualização
df_results = pd.DataFrame(
    {
        "real": y_test,
        "mediana": y_pred_median,
        "inferior": y_pred_lower,
        "superior": y_pred_upper,
    }
)

# Ordenando para visualização mais limpa
df_results["interval_size"] = df_results["superior"] - df_results["inferior"]
df_results = df_results.sort_values(by="real")

# Criando o gráfico de dispersão com intervalos de confiança
fig = px.scatter(
    df_results,
    x="real",
    y="mediana",
    title="Predições vs. Valores Reais",
    labels={"Valor Real": "real", "mediana": "mediana"},
)

# Adicionando a faixa de intervalo de confiança (5%-95%)
fig.add_traces(
    [
        dict(
            x=df_results["real"],
            y=df_results["inferior"],
            mode="lines",
            line=dict(width=0.5, color="rgba(0,100,200,0.3)"),
            name="Intervalo Inferior (5%)",
        ),
        dict(
            x=df_results["real"],
            y=df_results["superior"],
            mode="lines",
            line=dict(width=0.5, color="rgba(0,100,200,0.3)"),
            name="Intervalo Superior (95%)",
            fill="tonexty",  # Preenchendo entre as faixas
            fillcolor="rgba(0,100,200,0.1)",
        ),
    ]
)

# Adicionando a linha de predição perfeita (diagonal)
fig.add_traces(
    [
        dict(
            x=[min(df_results["real"]), max(df_results["real"])],
            y=[min(df_results["real"]), max(df_results["real"])],
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name="Predição Perfeita",
        )
    ]
)

# Exibindo o gráfico
fig.show()

breakpoint()
