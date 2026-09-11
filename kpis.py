import pandas as pd

def calcular_kpis_smart_grid(df):
    kpis_list = []

    for regiao, grupo in df.groupby('regiao'):
        c_medio = grupo['consumo_kwh'].mean()
        pico_demanda = grupo['consumo_kwh'].max()

        fator_carga = (c_medio / pico_demanda) if pico_demanda > 0 else 0
        anomalias = grupo['anomalia_sensor'].sum()
        taxa_anomalias = (anomalias / len(grupo)) * 100
        t_falha = grupo['status_falha'].mean() * 100

        kpis_list.append({
            'Região': regiao,
            'Consumo Médio (kWh)': round(c_medio, 2),
            'Pico de Demanda (kWh)': round(pico_demanda, 2),
            'Fator de Carga (Eficiência)': round(fator_carga, 2),
            'Taxa de Anomalias (%)': round(taxa_anomalias, 2),
            'Taxa de Falha (%)': round(t_falha, 2),
        })

    df_kpis = pd.DataFrame(kpis_list).sort_values(by='Consumo Médio (kWh)', ascending=False)
    return df_kpis