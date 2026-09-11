import pandas as pd
import numpy as np
import os


def executar_limpeza(caminho_entrada='data/raw/cidade_alfa_sensores.csv',
                     caminho_saida='data/processed/cidade_alfa_sensores_limpo.csv'):
    np.random.seed(42)

    # Garante que os diretórios de saída e entrada existam para evitar o erro de diretório inexistente
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    os.makedirs(os.path.dirname(caminho_entrada), exist_ok=True)

    try:
        df = pd.read_csv(caminho_entrada)
        print('>>> Dataset real carregado com sucesso!')
    except FileNotFoundError:
        print('>>> Arquivo não encontrado. Gerando base sintética de alta fidelidade...')
        regioes = ['Centro', 'Norte', 'Sul', 'Leste', 'Oeste']
        dias = 7
        intervalo_h = 2
        n_timestamps = (24 // intervalo_h) * dias

        timestamps = pd.date_range(start='2026-08-01', periods=n_timestamps, freq=f'{intervalo_h}h')
        df = pd.DataFrame([{'timestamp': ts, 'regiao': r} for ts in timestamps for r in regioes])
        n = len(df)

        df['consumo_kwh'] = np.random.normal(loc=160, scale=35, size=n)
        df['tensao_v'] = np.random.normal(loc=220, scale=4, size=n)
        df['status_falha'] = np.random.choice([0, 1], n, p=[0.92, 0.08])

        df.loc[12, 'consumo_kwh'] = np.nan
        df.loc[45, 'tensao_v'] = -45.0
        df.loc[88, 'consumo_kwh'] = 4500.0

    # A) Duplicatas
    df = df.drop_duplicates()

    # B) Valores Nulos
    if df['consumo_kwh'].isnull().sum() > 0:
        df['consumo_kwh'] = df['consumo_kwh'].fillna(
            df.groupby('regiao')['consumo_kwh'].transform('median')
        )

    # C) Detecção de Anomalias (Flags)
    cond_tensao_anomala = (df['tensao_v'] < 190) | (df['tensao_v'] > 250)
    media_c = df['consumo_kwh'].mean()
    std_c = df['consumo_kwh'].std()
    limite_sup = media_c + (4 * std_c)
    cond_consumo_anomalo = (df['consumo_kwh'] < 0) | (df['consumo_kwh'] > limite_sup)

    df['anomalia_sensor'] = (cond_tensao_anomala | cond_consumo_anomalo).astype(int)

    # D) Correção física
    df.loc[cond_tensao_anomala, 'tensao_v'] = 220.0
    df.loc[df['consumo_kwh'] > limite_sup, 'consumo_kwh'] = limite_sup
    df.loc[df['consumo_kwh'] < 0, 'consumo_kwh'] = df.groupby('regiao')['consumo_kwh'].transform('median')

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Salvar processado
    df.to_csv(caminho_saida, index=False)
    print(f'>>> Base limpa e salva com sucesso em {caminho_saida}')
    return df