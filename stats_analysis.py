# ==============================================================================
# stats_analysis.py
# CAMADA DE ESTATÍSTICA INFERENCIAL E PREDITIVA — PROJETO CIDADE ALFA
# Testes de Hipótese | Análise de Correlação | Estatística Preditiva
# ==============================================================================

import pandas as pd
import numpy as np
from scipy import stats


# ------------------------------------------------------------------------------
# 1. TESTE DE HIPÓTESE — Consumo difere entre regiões?
# ------------------------------------------------------------------------------
def teste_hipotese_consumo_regioes(df, alpha=0.05):
    """
    H0: O consumo médio de energia é estatisticamente igual entre todas as regiões.
    H1: Existe pelo menos uma região com consumo médio diferente das demais.

    Usa ANOVA one-way (Teste F) por comparar mais de 2 grupos simultaneamente.
    Se significativo, roda testes t pareados (post-hoc) para identificar QUAIS
    regiões diferem entre si.
    """
    grupos = [grupo['consumo_kwh'].values for _, grupo in df.groupby('regiao')]
    regioes = df['regiao'].unique().tolist()

    f_stat, p_valor = stats.f_oneway(*grupos)

    resultado = {
        'teste': 'ANOVA one-way',
        'hipotese_nula': 'Consumo médio é igual entre todas as regiões',
        'estatistica_F': round(f_stat, 4),
        'p_valor': round(p_valor, 6),
        'alpha': alpha,
        'rejeita_h0': bool(p_valor < alpha),
        'conclusao': (
            'Há diferença estatisticamente significativa no consumo entre regiões.'
            if p_valor < alpha else
            'Não há evidência suficiente de diferença no consumo entre regiões.'
        ),
    }

    print('--- [TESTE DE HIPÓTESE] Consumo de Energia entre Regiões (ANOVA) ---')
    print(f"F-estatística: {resultado['estatistica_F']} | p-valor: {resultado['p_valor']}")
    print(f">>> {resultado['conclusao']}")

    # Post-hoc: comparações par a par (apenas se ANOVA for significativa)
    pares_significativos = []
    if resultado['rejeita_h0']:
        print('\n--- Comparações Par a Par (Teste t de Welch, post-hoc) ---')
        for i in range(len(regioes)):
            for j in range(i + 1, len(regioes)):
                r1, r2 = regioes[i], regioes[j]
                g1 = df[df['regiao'] == r1]['consumo_kwh']
                g2 = df[df['regiao'] == r2]['consumo_kwh']
                t_stat, p_par = stats.ttest_ind(g1, g2, equal_var=False)
                significativo = p_par < alpha
                if significativo:
                    pares_significativos.append((r1, r2, round(p_par, 4)))
                print(f"{r1} vs {r2}: p-valor={p_par:.4f} "
                      f"{'(diferença significativa)' if significativo else '(sem diferença significativa)'}")

    resultado['pares_significativos'] = pares_significativos
    return resultado


# ------------------------------------------------------------------------------
# 2. ANÁLISE DE CORRELAÇÃO — quais variáveis se relacionam?
# ------------------------------------------------------------------------------
def analise_correlacao(df, alpha=0.05):
    """
    Investiga correlação entre variáveis numéricas e binárias do sistema elétrico:
    consumo_kwh, tensao_v, anomalia_sensor, status_falha.

    Usa Pearson para todos os pares (válido também para variáveis binárias,
    equivalente à correlação ponto-bisserial quando uma variável é 0/1).
    Reporta coeficiente + p-valor de significância para cada par.
    """
    variaveis = ['consumo_kwh', 'tensao_v', 'anomalia_sensor', 'status_falha']
    variaveis = [v for v in variaveis if v in df.columns]

    matriz_corr = df[variaveis].corr(method='pearson')

    print('--- [ANÁLISE DE CORRELAÇÃO] Matriz de Correlação (Pearson) ---')
    print(matriz_corr.round(3))

    resultados_pares = []
    print('\n--- Significância estatística por par de variáveis ---')
    for i in range(len(variaveis)):
        for j in range(i + 1, len(variaveis)):
            v1, v2 = variaveis[i], variaveis[j]
            r, p_valor = stats.pearsonr(df[v1], df[v2])
            significativo = p_valor < alpha

            if abs(r) < 0.1:
                forca = 'desprezível'
            elif abs(r) < 0.3:
                forca = 'fraca'
            elif abs(r) < 0.5:
                forca = 'moderada'
            else:
                forca = 'forte'

            resultados_pares.append({
                'variavel_1': v1, 'variavel_2': v2,
                'correlacao_r': round(r, 3), 'p_valor': round(p_valor, 6),
                'forca': forca, 'significativo': significativo,
            })
            print(f"{v1} × {v2}: r={r:.3f} ({forca}) | p-valor={p_valor:.4f} "
                  f"{'(significativo)' if significativo else '(não significativo)'}")

    return pd.DataFrame(resultados_pares), matriz_corr


# ------------------------------------------------------------------------------
# 3. ESTATÍSTICA PREDITIVA — projeção de consumo futuro por região
# ------------------------------------------------------------------------------
def prever_consumo_futuro(df, regiao, periodos_futuros=12, intervalo_h=2):
    """
    Regressão linear simples sobre a série temporal de consumo de uma região,
    usada para projetar tendência de curto prazo (apoio à decisão operacional).

    Retorna: DataFrame com timestamps futuros e consumo projetado,
    além do R² do ajuste (qualidade da tendência linear).
    """
    df_regiao = df[df['regiao'] == regiao].sort_values('timestamp').reset_index(drop=True)

    if len(df_regiao) < 3:
        raise ValueError(f"Dados insuficientes para prever consumo da região '{regiao}'.")

    # Variável independente = índice temporal sequencial (não a data bruta)
    x = np.arange(len(df_regiao))
    y = df_regiao['consumo_kwh'].values

    slope, intercept, r_valor, p_valor, std_err = stats.linregress(x, y)

    # Projeção para os próximos períodos
    x_futuro = np.arange(len(df_regiao), len(df_regiao) + periodos_futuros)
    y_previsto = slope * x_futuro + intercept

    ultimo_timestamp = df_regiao['timestamp'].iloc[-1]
    timestamps_futuros = pd.date_range(
        start=ultimo_timestamp, periods=periodos_futuros + 1, freq=f'{intervalo_h}h'
    )[1:]

    df_previsao = pd.DataFrame({
        'timestamp': timestamps_futuros,
        'regiao': regiao,
        'consumo_previsto_kwh': np.round(y_previsto, 2),
    })

    r_quadrado = r_valor ** 2
    tendencia = 'crescente' if slope > 0 else 'decrescente' if slope < 0 else 'estável'

    print(f"--- [PREVISÃO] Consumo Futuro — Região: {regiao} ---")
    print(f"Tendência: {tendencia} | Inclinação: {slope:.3f} kWh/leitura | R²: {r_quadrado:.3f}")
    if r_quadrado < 0.3:
        print(">>> Aviso: ajuste linear fraco — a tendência não é claramente linear "
              "(considere isso ao comunicar a previsão como estimativa aproximada).")

    return df_previsao, {'slope': slope, 'intercept': intercept, 'r2': r_quadrado, 'tendencia': tendencia}


# ------------------------------------------------------------------------------
# Execução direta para teste isolado do módulo
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    from preprocessing import executar_limpeza

    df = executar_limpeza()

    print('\n' + '=' * 70)
    resultado_anova = teste_hipotese_consumo_regioes(df)

    print('\n' + '=' * 70)
    df_correlacoes, matriz = analise_correlacao(df)

    print('\n' + '=' * 70)
    regiao_teste = df['regiao'].unique()[0]
    df_previsao, metricas = prever_consumo_futuro(df, regiao=regiao_teste)
    print(df_previsao)