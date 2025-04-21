import argparse
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=00)

# Treinando o RandomForestQuantileRegressor
model = RandomForestQuantileRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Fazendo previsões para o conjunto de teste
quantiles = [0.10, 0.5, 0.90]  # Inferior, mediana e superior
predictions = model.predict(X_test, quantiles=quantiles)

# Criando DataFrame com previsões
results_df = pd.DataFrame(
    {
        "y_true": y_test.values,
        "y_pred_low": predictions[:, 0],
        "y_pred_med": predictions[:, 1],
        "y_pred_upp": predictions[:, 2],
    }
)

# Ordenar pelo valor real de y
results_df_sorted = results_df.sort_values(by="y_true").reset_index(drop=True)

# Criando figura
fig = go.Figure()

# Faixa de confiança (como área sombreada)
fig.add_trace(
    go.Scatter(
        x=results_df_sorted["y_true"],
        y=results_df_sorted["y_pred_upp"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=results_df_sorted["y_true"],
        y=results_df_sorted["y_pred_low"],
        fill="tonexty",
        fillcolor="rgba(150,150,150,0.3)",
        line=dict(width=0),
        mode="lines",
        name="Intervalo 90%",
        hoverinfo="skip",
    )
)

# Mediana predita
fig.add_trace(
    go.Scatter(
        x=results_df_sorted["y_true"],
        y=results_df_sorted["y_pred_med"],
        mode="markers",
        name="Previsão (mediana)",
        marker=dict(size=6, color="royalblue"),
    )
)

# Reta y = x (previsão ideal)
min_val = min(results_df_sorted["y_true"].min(), results_df_sorted["y_pred_med"].min())
max_val = max(results_df_sorted["y_true"].max(), results_df_sorted["y_pred_med"].max())
fig.add_trace(
    go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="black", dash="dash"),
        name="Previsão ideal (y = x)",
    )
)

# Layout
fig.update_layout(
    title="Previsão vs Valor Real com Intervalos de Confiança",
    xaxis_title="Valor real (Y)",
    yaxis_title="Previsão (mediana)",
    template="simple_white",
)

# Salvando
fig.write_html("viz/qrf/qrf_intervalos.html")
