# ==============================================================================
# clustering.py
# MINERAÇÃO DE DADOS — PROJETO CIDADE ALFA
# Clustering de perfis de consumo elétrico via K-Means
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# Colunas usadas para caracterizar o "perfil" de cada leitura de sensor
FEATURES = ["consumo_kwh", "tensao_v", "hora"]


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante que a coluna 'hora' existe (derivada do timestamp) e seleciona
    apenas as colunas numéricas usadas como características do clustering.
    """
    df = df.copy()
    if "hora" not in df.columns:
        df["hora"] = pd.to_datetime(df["timestamp"]).dt.hour
    return df


def encontrar_k_ideal(df: pd.DataFrame, k_min: int = 2, k_max: int = 8) -> pd.DataFrame:
    """
    Testa diferentes valores de k (número de clusters) e calcula o Silhouette
    Score para cada um — métrica que mede o quão bem separados e coesos estão
    os clusters (varia de -1 a 1; quanto maior, melhor).

    Usada para justificar a escolha do k final de forma objetiva, em vez de
    "chutar" um número de clusters (é o que um avaliador de estatística
    espera ver: escolha de hiperparâmetro fundamentada em métrica, não em
    achismo).
    """
    df_prep = preparar_features(df)
    X = df_prep[FEATURES].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    resultados = []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        resultados.append({"k": k, "silhouette_score": round(score, 4)})

    df_resultados = pd.DataFrame(resultados)
    print("--- [CLUSTERING] Silhouette Score por número de clusters (k) ---")
    print(df_resultados)

    melhor_k = df_resultados.loc[df_resultados["silhouette_score"].idxmax(), "k"]
    print(f">>> Melhor k sugerido: {int(melhor_k)} (maior Silhouette Score)")

    return df_resultados


def aplicar_clustering(df: pd.DataFrame, n_clusters: int = 3) -> tuple[pd.DataFrame, dict]:
    """
    Aplica K-Means com o número de clusters definido, padronizando as features
    antes (StandardScaler é obrigatório aqui: consumo está em kWh na casa das
    centenas, hora vai de 0 a 23 — sem padronizar, consumo dominaria a métrica
    de distância e a hora seria praticamente ignorada).

    Retorna:
        - DataFrame original com nova coluna 'cluster'
        - dict com metadados do modelo (score, centróides, perfil por cluster)
    """
    df_prep = preparar_features(df)
    X = df_prep[FEATURES].dropna()
    indices_validos = X.index

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    df_prep = df_prep.loc[indices_validos].copy()
    df_prep["cluster"] = labels

    score = silhouette_score(X_scaled, labels)

    # Perfil médio de cada cluster — essencial para NOMEAR os clusters com
    # significado de negócio (ex: "Cluster 0 = alto consumo noturno")
    perfil_clusters = df_prep.groupby("cluster")[FEATURES + ["anomalia_sensor", "status_falha"]].mean().round(2)

    print(f"--- [CLUSTERING] K-Means aplicado com k={n_clusters} ---")
    print(f"Silhouette Score: {score:.4f}")
    print("\nPerfil médio por cluster:")
    print(perfil_clusters)

    metadados = {
        "n_clusters": n_clusters,
        "silhouette_score": round(score, 4),
        "perfil_clusters": perfil_clusters,
        "centroides_escala_original": scaler.inverse_transform(kmeans.cluster_centers_),
    }

    return df_prep, metadados


def nomear_clusters(perfil_clusters: pd.DataFrame) -> dict:
    """
    Gera rótulos descritivos automáticos para cada cluster, com base no
    consumo médio relativo e na hora média — traduz números em linguagem
    de negócio para apresentação à gestão pública (em vez de expor
    "Cluster 0", "Cluster 1" sem contexto).
    """
    consumo_mediano = perfil_clusters["consumo_kwh"].median()
    rotulos = {}

    for cluster_id, linha in perfil_clusters.iterrows():
        nivel_consumo = "Alto Consumo" if linha["consumo_kwh"] > consumo_mediano else "Baixo Consumo"
        periodo = "Noturno" if (linha["hora"] >= 19 or linha["hora"] <= 5) else "Diurno"
        risco = " (Atenção: alta taxa de anomalia)" if linha["anomalia_sensor"] > 0.1 else ""
        rotulos[cluster_id] = f"{nivel_consumo} — {periodo}{risco}"

    return rotulos


# ------------------------------------------------------------------------------
# Execução direta para teste isolado do módulo
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    from preprocessing import executar_limpeza

    df = executar_limpeza()

    print("=" * 70)
    encontrar_k_ideal(df)

    print("\n" + "=" * 70)
    df_clusterizado, metadados = aplicar_clustering(df, n_clusters=3)

    print("\n" + "=" * 70)
    rotulos = nomear_clusters(metadados["perfil_clusters"])
    print("--- Rótulos de negócio por cluster ---")
    for cluster_id, rotulo in rotulos.items():
        print(f"Cluster {cluster_id}: {rotulo}")