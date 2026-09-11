# ==============================================================================
# sql_layer.py
# CAMADA SQL ANALÍTICA — PROJETO CIDADE ALFA
# Consultas analíticas via DuckDB (agregações, window functions, ranking)
# ==============================================================================

import duckdb
import pandas as pd


def conectar(df: pd.DataFrame):
    """
    Cria uma conexão DuckDB em memória e registra o DataFrame como tabela SQL.
    DuckDB permite consultar DataFrames Pandas diretamente via SQL, sem ETL
    intermediário — ideal para uma camada analítica leve sobre dados já limpos.
    """
    con = duckdb.connect(database=':memory:')
    con.register('sensores', df)
    return con


# ------------------------------------------------------------------------------
# 1. Consumo médio, mínimo e máximo por região (agregação básica)
# ------------------------------------------------------------------------------
def query_resumo_por_regiao(df: pd.DataFrame) -> pd.DataFrame:
    con = conectar(df)
    query = """
        SELECT
            regiao,
            COUNT(*)                       AS total_leituras,
            ROUND(AVG(consumo_kwh), 2)     AS consumo_medio_kwh,
            ROUND(MIN(consumo_kwh), 2)     AS consumo_minimo_kwh,
            ROUND(MAX(consumo_kwh), 2)     AS consumo_maximo_kwh,
            ROUND(STDDEV(consumo_kwh), 2)  AS desvio_padrao_kwh,
            SUM(anomalia_sensor)           AS total_anomalias,
            SUM(status_falha)              AS total_falhas
        FROM sensores
        GROUP BY regiao
        ORDER BY consumo_medio_kwh DESC;
    """
    resultado = con.execute(query).fetchdf()
    con.close()
    return resultado


# ------------------------------------------------------------------------------
# 2. Ranking de horários de pico por região (WINDOW FUNCTION)
# ------------------------------------------------------------------------------
def query_ranking_picos_por_regiao(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Usa RANK() OVER (PARTITION BY ...) para identificar, dentro de cada região,
    quais foram os horários (timestamps) de maior consumo registrado —
    útil para a gestão pública priorizar janelas de monitoramento de carga.
    """
    con = conectar(df)
    query = f"""
        WITH ranking AS (
            SELECT
                regiao,
                timestamp,
                consumo_kwh,
                RANK() OVER (PARTITION BY regiao ORDER BY consumo_kwh DESC) AS posicao
            FROM sensores
        )
        SELECT regiao, timestamp, ROUND(consumo_kwh, 2) AS consumo_kwh, posicao
        FROM ranking
        WHERE posicao <= {top_n}
        ORDER BY regiao, posicao;
    """
    resultado = con.execute(query).fetchdf()
    con.close()
    return resultado


# ------------------------------------------------------------------------------
# 3. Média móvel de consumo (WINDOW FUNCTION — suavização de tendência)
# ------------------------------------------------------------------------------
def query_media_movel_consumo(df: pd.DataFrame, janela: int = 4) -> pd.DataFrame:
    """
    Calcula média móvel de consumo por região usando AVG(...) OVER com janela
    deslizante — técnica clássica de análise de séries temporais em SQL,
    complementar à regressão linear feita em stats_analysis.py.
    """
    con = conectar(df)
    query = f"""
        SELECT
            regiao,
            timestamp,
            ROUND(consumo_kwh, 2) AS consumo_kwh,
            ROUND(AVG(consumo_kwh) OVER (
                PARTITION BY regiao
                ORDER BY timestamp
                ROWS BETWEEN {janela - 1} PRECEDING AND CURRENT ROW
            ), 2) AS media_movel_kwh
        FROM sensores
        ORDER BY regiao, timestamp;
    """
    resultado = con.execute(query).fetchdf()
    con.close()
    return resultado


# ------------------------------------------------------------------------------
# 4. Distribuição de anomalias e falhas por faixa horária
# ------------------------------------------------------------------------------
def query_anomalias_por_faixa_horaria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifica cada leitura em faixa horária (Madrugada/Manhã/Tarde/Noite) via
    CASE WHEN e agrega taxa de anomalia — identifica se falhas se concentram
    em horários específicos (ex: picos noturnos de consumo residencial).
    """
    con = conectar(df)
    query = """
        SELECT
            regiao,
            CASE
                WHEN EXTRACT(HOUR FROM timestamp) BETWEEN 0 AND 5   THEN 'Madrugada'
                WHEN EXTRACT(HOUR FROM timestamp) BETWEEN 6 AND 11  THEN 'Manhã'
                WHEN EXTRACT(HOUR FROM timestamp) BETWEEN 12 AND 17 THEN 'Tarde'
                ELSE 'Noite'
            END AS faixa_horaria,
            COUNT(*)                                   AS total_leituras,
            SUM(anomalia_sensor)                        AS total_anomalias,
            ROUND(100.0 * SUM(anomalia_sensor) / COUNT(*), 2) AS taxa_anomalia_pct
        FROM sensores
        GROUP BY regiao, faixa_horaria
        ORDER BY regiao,
            CASE faixa_horaria
                WHEN 'Madrugada' THEN 1 WHEN 'Manhã' THEN 2
                WHEN 'Tarde' THEN 3 ELSE 4
            END;
    """
    resultado = con.execute(query).fetchdf()
    con.close()
    return resultado


# ------------------------------------------------------------------------------
# 5. Regiões acima da média geral de consumo (subquery correlacionada)
# ------------------------------------------------------------------------------
def query_regioes_acima_da_media(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demonstra subquery escalar — regiões cujo consumo médio supera a média
    geral da cidade, sinalizando prioridade de intervenção.
    """
    con = conectar(df)
    query = """
        SELECT
            regiao,
            ROUND(AVG(consumo_kwh), 2) AS consumo_medio_regiao,
            (SELECT ROUND(AVG(consumo_kwh), 2) FROM sensores) AS consumo_medio_cidade
        FROM sensores
        GROUP BY regiao
        HAVING AVG(consumo_kwh) > (SELECT AVG(consumo_kwh) FROM sensores)
        ORDER BY consumo_medio_regiao DESC;
    """
    resultado = con.execute(query).fetchdf()
    con.close()
    return resultado


# ------------------------------------------------------------------------------
# Execução direta para teste isolado do módulo
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from preprocessing import executar_limpeza

    df = executar_limpeza()

    print('\n=== [SQL] Resumo por Região ===')
    print(query_resumo_por_regiao(df))

    print('\n=== [SQL] Top 3 Picos de Consumo por Região (Window Function) ===')
    print(query_ranking_picos_por_regiao(df))

    print('\n=== [SQL] Média Móvel de Consumo (janela=4) ===')
    print(query_media_movel_consumo(df).head(10))

    print('\n=== [SQL] Anomalias por Faixa Horária ===')
    print(query_anomalias_por_faixa_horaria(df))

    print('\n=== [SQL] Regiões Acima da Média Geral ===')
    print(query_regioes_acima_da_media(df))