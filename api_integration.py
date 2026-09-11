# ==============================================================================
# api_integration.py
# INTEGRAÇÃO COM API EXTERNA — PROJETO CIDADE ALFA
# Consumo da API Open-Meteo (clima) para enriquecer a análise de consumo elétrico
# ==============================================================================

import requests
import pandas as pd
from datetime import datetime, timedelta


# URL base da API Open-Meteo — gratuita, sem necessidade de API key
BASE_URL = "https://api.open-meteo.com/v1/forecast"

# Coordenadas de referência (São Paulo, usada como proxy para a "Cidade Alfa")
LATITUDE = -23.5505
LONGITUDE = -46.6333


def buscar_dados_climaticos(data_inicio: str, data_fim: str, timeout: int = 10) -> pd.DataFrame:
    """
    Busca temperatura horária histórica via API Open-Meteo para o período informado.

    Parâmetros:
        data_inicio, data_fim: strings no formato 'YYYY-MM-DD'
        timeout: tempo máximo de espera pela resposta da API (segundos)

    Retorna:
        DataFrame com colunas ['timestamp', 'temperatura_c'], ou DataFrame vazio
        em caso de falha (a função NUNCA lança exceção não tratada — é resiliente
        a falhas de rede, comuns em ambiente acadêmico/apresentação ao vivo).
    """
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": data_inicio,
        "end_date": data_fim,
        "hourly": "temperature_2m",
        "timezone": "America/Sao_Paulo",
    }

    try:
        resposta = requests.get(BASE_URL, params=params, timeout=timeout)
        resposta.raise_for_status()  # lança erro para status HTTP 4xx/5xx
        dados = resposta.json()

        df_clima = pd.DataFrame({
            "timestamp": pd.to_datetime(dados["hourly"]["time"]),
            "temperatura_c": dados["hourly"]["temperature_2m"],
        })

        print(f">>> [API] Dados climáticos obtidos com sucesso: {len(df_clima)} registros horários.")
        return df_clima

    except requests.exceptions.Timeout:
        print(">>> [API] Aviso: tempo limite excedido ao consultar a API de clima.")
    except requests.exceptions.ConnectionError:
        print(">>> [API] Aviso: falha de conexão com a API de clima (verifique sua internet).")
    except requests.exceptions.HTTPError as e:
        print(f">>> [API] Aviso: erro HTTP retornado pela API de clima: {e}")
    except (KeyError, ValueError) as e:
        print(f">>> [API] Aviso: resposta da API em formato inesperado: {e}")

    # Fallback: retorna DataFrame vazio com o schema esperado, para não quebrar
    # o pipeline downstream (quem consumir isso deve checar se está vazio)
    return pd.DataFrame(columns=["timestamp", "temperatura_c"])


def enriquecer_com_clima(df_sensores: pd.DataFrame) -> pd.DataFrame:
    """
    Integra os dados de clima ao DataFrame de sensores elétricos, casando por
    timestamp mais próximo (merge_asof), já que os sensores registram a cada
    2h e a API retorna dados horários — os índices não coincidem exatamente.

    Se a API falhar, retorna o DataFrame original com a coluna 'temperatura_c'
    preenchida com NaN, para que o restante do pipeline continue funcionando.
    """
    data_inicio = df_sensores["timestamp"].min().strftime("%Y-%m-%d")
    data_fim = df_sensores["timestamp"].max().strftime("%Y-%m-%d")

    df_clima = buscar_dados_climaticos(data_inicio, data_fim)

    if df_clima.empty:
        print(">>> [API] Prosseguindo sem dados climáticos (coluna preenchida como NaN).")
        df_sensores = df_sensores.copy()
        df_sensores["temperatura_c"] = pd.NA
        return df_sensores

    # merge_asof exige ambos os DataFrames ordenados pela chave temporal
    df_sensores_ordenado = df_sensores.sort_values("timestamp").reset_index(drop=True)
    df_clima_ordenado = df_clima.sort_values("timestamp").reset_index(drop=True)

    df_enriquecido = pd.merge_asof(
        df_sensores_ordenado,
        df_clima_ordenado,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("1h"),
    )

    n_sem_clima = df_enriquecido["temperatura_c"].isna().sum()
    if n_sem_clima > 0:
        print(f">>> [API] Aviso: {n_sem_clima} leituras não encontraram temperatura correspondente (fora da tolerância de 1h).")

    print(">>> [API] Dados de sensores enriquecidos com temperatura com sucesso.")
    return df_enriquecido


# ------------------------------------------------------------------------------
# Execução direta para teste isolado do módulo
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    from preprocessing import executar_limpeza

    df = executar_limpeza()
    df_enriquecido = enriquecer_com_clima(df)

    print("\n--- Amostra do dataset enriquecido ---")
    print(df_enriquecido[["timestamp", "regiao", "consumo_kwh", "temperatura_c"]].head(10))

    # Correlação rápida entre temperatura e consumo (validação exploratória)
    if df_enriquecido["temperatura_c"].notna().sum() > 2:
        correlacao = df_enriquecido[["consumo_kwh", "temperatura_c"]].corr().iloc[0, 1]
        print(f"\n>>> Correlação exploratória consumo × temperatura: {correlacao:.3f}")