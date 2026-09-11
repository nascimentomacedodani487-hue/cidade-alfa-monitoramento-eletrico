import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from preprocessing import executar_limpeza
from kpis import calcular_kpis_smart_grid
from stats_analysis import (
    teste_hipotese_consumo_regioes,
    analise_correlacao,
    prever_consumo_futuro,
)
from sql_layer import (
    query_resumo_por_regiao,
    query_ranking_picos_por_regiao,
    query_media_movel_consumo,
    query_anomalias_por_faixa_horaria,
)
from clustering import aplicar_clustering, nomear_clusters
from api_integration import enriquecer_com_clima

# ==============================================================================
# CONFIGURAÇÃO GLOBAL DE ESTILO
# ==============================================================================

# Paleta fixa por região — garante que "Centro" seja sempre a mesma cor
# em todos os gráficos do dashboard, em todas as abas
CORES_REGIAO = {
    'Centro': '#2E86AB',
    'Norte': '#A23B72',
    'Sul': '#F18F01',
    'Leste': '#3B8686',
    'Oeste': '#C73E1D',
}
CORES_CLUSTER = px.colors.qualitative.Set2
TEMPLATE_PADRAO = 'plotly_white'

st.set_page_config(
    page_title="Cidade Alfa - Monitoramento Inteligente",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Projeto Cidade Alfa: Smart Grid Dashboard")
st.markdown("Painel interativo integrando engenharia de dados, estatística inferencial, mineração de dados e monitoramento em tempo real.")


@st.cache_data
def carregar_ou_processar_dados():
    try:
        df = pd.read_csv('data/processed/cidade_alfa_sensores_limpo.csv')
    except FileNotFoundError:
        df = executar_limpeza(
            caminho_entrada='data/raw/cidade_alfa_sensores.csv',
            caminho_saida='data/processed/cidade_alfa_sensores_limpo.csv'
        )
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['data_dia'] = df['timestamp'].dt.date
    df['hora'] = df['timestamp'].dt.hour
    return df


df_base = carregar_ou_processar_dados()

# ------------------------------------------------------------------------------
# SIMULAÇÃO DE TEMPO REAL
# ------------------------------------------------------------------------------
def gerar_nova_leitura(regioes: list) -> dict:
    """Gera uma leitura sintética plausível para o instante atual."""
    regiao = np.random.choice(regioes)
    consumo = max(np.random.normal(loc=160, scale=35), 0)
    tensao = np.random.normal(loc=220, scale=4)
    anomalia = int((tensao < 190) or (tensao > 250) or (consumo > 300))
    falha = int(np.random.choice([0, 1], p=[0.92, 0.08]))
    agora = pd.Timestamp.now()
    return {
        'timestamp': agora, 'regiao': regiao,
        'consumo_kwh': round(consumo, 2), 'tensao_v': round(tensao, 2),
        'status_falha': falha, 'anomalia_sensor': anomalia,
        'data_dia': agora.date(), 'hora': agora.hour,
    }


if 'buffer_tempo_real' not in st.session_state:
    st.session_state.buffer_tempo_real = df_base.iloc[0:0].copy()

# --- SIDEBAR (Filtros) ---
st.sidebar.header("Filtros Globais")

regioes_todas = df_base['regiao'].unique().tolist()

if 'regioes_filtro' not in st.session_state:
    st.session_state.regioes_filtro = regioes_todas.copy()

col_sel1, col_sel2 = st.sidebar.columns(2)
if col_sel1.button("Selecionar todas", use_container_width=True):
    st.session_state.regioes_filtro = regioes_todas.copy()
if col_sel2.button("Limpar seleção", use_container_width=True):
    st.session_state.regioes_filtro = []

regiao_selecionada = st.sidebar.multiselect(
    "Selecione as Regiões:",
    options=regioes_todas,
    key='regioes_filtro',
)

# Combina histórico + buffer de tempo real
df = pd.concat([df_base, st.session_state.buffer_tempo_real], ignore_index=True)

# Blindagem de tipos pós-concat (evita bug de dtype 'object' em gráficos)
df['consumo_kwh'] = pd.to_numeric(df['consumo_kwh'], errors='coerce')
df['tensao_v'] = pd.to_numeric(df['tensao_v'], errors='coerce')
df['anomalia_sensor'] = pd.to_numeric(df['anomalia_sensor'], errors='coerce').fillna(0).astype(int)
df['status_falha'] = pd.to_numeric(df['status_falha'], errors='coerce').fillna(0).astype(int)
df['hora'] = pd.to_numeric(df['hora'], errors='coerce').astype(int)

df_filtrado = df[df['regiao'].isin(regiao_selecionada)]

if df_filtrado.empty:
    st.warning("Por favor, selecione pelo menos uma região na barra lateral.")
    st.stop()

# --- MÉTRICAS PRINCIPAIS ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total de Leituras", len(df_filtrado))
col2.metric("Consumo Médio Geral", f"{df_filtrado['consumo_kwh'].mean():.2f} kWh")
col3.metric("Anomalias Detectadas", int(df_filtrado['anomalia_sensor'].sum()))
col4.metric("Taxa Média de Falha", f"{df_filtrado['status_falha'].mean() * 100:.1f}%")
col5.metric("Leituras em Tempo Real", len(st.session_state.buffer_tempo_real))

st.markdown("---")

tab_kpis, tab_visual, tab_stats, tab_previsao, tab_sql, tab_mineracao, tab_clima, tab_tempo_real, tab_recomendacoes = st.tabs([
    "📊 KPIs", "📈 Visualização", "🔬 Testes Estatísticos", "🔮 Previsão",
    "🗄️ SQL Analítico", "🧩 Mineração de Dados", "🌡️ Clima", "⏱️ Tempo Real", "📝 Recomendações",
])

# --- KPIs ---
with tab_kpis:
    st.subheader("Indicadores de Desempenho (Smart Grid) por Região")
    st.dataframe(calcular_kpis_smart_grid(df_filtrado), use_container_width=True)

# --- VISUALIZAÇÃO ---
with tab_visual:
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        resumo_regiao = df_filtrado.groupby('regiao')['consumo_kwh'].agg(['mean', 'std']).reset_index()
        fig_bar = px.bar(
            resumo_regiao, x='regiao', y='mean', error_y='std',
            color='regiao', color_discrete_map=CORES_REGIAO, template=TEMPLATE_PADRAO,
            title='Consumo Médio de Energia por Região',
            labels={'mean': 'Consumo Médio (kWh)', 'regiao': 'Região'},
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_g2:
        df_temporal = df_filtrado.groupby(['data_dia', 'regiao'])['consumo_kwh'].mean().reset_index()
        df_temporal['data_dia'] = pd.to_datetime(df_temporal['data_dia'])
        fig_line = px.line(
            df_temporal, x='data_dia', y='consumo_kwh', color='regiao', markers=True,
            color_discrete_map=CORES_REGIAO, template=TEMPLATE_PADRAO,
            title='Tendência Diária de Consumo Elétrico por Região',
            labels={'consumo_kwh': 'Consumo Médio Diário (kWh)', 'data_dia': 'Data'},
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("**Concentração de Anomalias por Região e Horário**")
    pivot_anomalias = df_filtrado.pivot_table(index='regiao', columns='hora', values='anomalia_sensor', aggfunc='mean')
    fig_heat = px.imshow(
        pivot_anomalias, color_continuous_scale='YlOrRd', aspect='auto', template=TEMPLATE_PADRAO,
        labels={'x': 'Hora do Dia', 'y': 'Região', 'color': 'Taxa de Anomalia'},
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# --- TESTES ESTATÍSTICOS ---
with tab_stats:
    st.subheader("Teste de Hipótese — Consumo entre Regiões (ANOVA)")
    if df_filtrado['regiao'].nunique() < 2:
        st.info("Selecione ao menos 2 regiões para rodar o teste de hipótese.")
    else:
        resultado_anova = teste_hipotese_consumo_regioes(df_filtrado)
        colA, colB = st.columns(2)
        colA.metric("Estatística F", resultado_anova['estatistica_F'])
        colB.metric("p-valor", resultado_anova['p_valor'])
        st.write(f"**Conclusão:** {resultado_anova['conclusao']}")
        for r1, r2, p in resultado_anova['pares_significativos']:
            st.write(f"- {r1} vs {r2} (p={p})")

        fig_box = px.box(
            df_filtrado, x='regiao', y='consumo_kwh', color='regiao',
            color_discrete_map=CORES_REGIAO, template=TEMPLATE_PADRAO,
            title='Distribuição de Consumo por Região (suporte visual à ANOVA)',
            labels={'consumo_kwh': 'Consumo (kWh)', 'regiao': 'Região'},
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    st.subheader("Análise de Correlação")
    df_correlacoes, matriz_corr = analise_correlacao(df_filtrado)
    fig_corr = px.imshow(
        matriz_corr, text_auto='.2f', color_continuous_scale='RdYlGn', zmin=-1, zmax=1,
        template=TEMPLATE_PADRAO, title='Matriz de Correlação (Pearson)',
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.dataframe(df_correlacoes, use_container_width=True)

# --- PREVISÃO ---
with tab_previsao:
    st.subheader("Projeção de Consumo Futuro (Regressão Linear)")
    regiao_previsao = st.selectbox("Escolha a região para prever:", regiao_selecionada)
    periodos = st.slider("Períodos futuros (intervalos de 2h):", 4, 48, 12, 4)
    try:
        df_previsao, metricas = prever_consumo_futuro(df_filtrado, regiao=regiao_previsao, periodos_futuros=periodos)
        st.write(f"**Tendência:** {metricas['tendencia']} | **R²:** {metricas['r2']:.3f}")
        if metricas['r2'] < 0.3:
            st.warning("Ajuste linear fraco — trate a previsão como estimativa aproximada.")

        df_hist = df_filtrado[df_filtrado['regiao'] == regiao_previsao][['timestamp', 'consumo_kwh']].rename(columns={'consumo_kwh': 'valor'})
        df_hist['tipo'] = 'Histórico'
        df_fut = df_previsao.rename(columns={'consumo_previsto_kwh': 'valor'})[['timestamp', 'valor']]
        df_fut['tipo'] = 'Previsto'
        df_plot = pd.concat([df_hist, df_fut])

        fig_prev = px.line(
            df_plot, x='timestamp', y='valor', color='tipo', markers=True,
            template=TEMPLATE_PADRAO,
            title=f'Consumo Histórico × Previsto — {regiao_previsao}',
            labels={'valor': 'Consumo (kWh)', 'timestamp': 'Data/Hora'},
            color_discrete_map={'Histórico': '#2E86AB', 'Previsto': '#F18F01'},
        )
        st.plotly_chart(fig_prev, use_container_width=True)
        st.dataframe(df_previsao, use_container_width=True)
    except ValueError as e:
        st.error(str(e))

# --- SQL ---
with tab_sql:
    st.subheader("Consultas Analíticas (DuckDB)")
    st.markdown("**Resumo por Região**")
    st.dataframe(query_resumo_por_regiao(df_filtrado), use_container_width=True)
    st.markdown("**Top 3 Picos de Consumo por Região**")
    st.dataframe(query_ranking_picos_por_regiao(df_filtrado), use_container_width=True)
    st.markdown("**Anomalias por Faixa Horária**")
    st.dataframe(query_anomalias_por_faixa_horaria(df_filtrado), use_container_width=True)
    with st.expander("Ver Média Móvel de Consumo (janela=4)"):
        st.dataframe(query_media_movel_consumo(df_filtrado), use_container_width=True)

# --- MINERAÇÃO DE DADOS ---
with tab_mineracao:
    st.subheader("Mineração de Dados — Perfis de Consumo (K-Means)")
    n_clusters = st.slider("Número de clusters (k):", 2, 6, 3)
    df_clusterizado, metadados = aplicar_clustering(df_filtrado, n_clusters=n_clusters)
    rotulos = nomear_clusters(metadados['perfil_clusters'])

    st.metric("Silhouette Score", metadados['silhouette_score'])
    st.dataframe(metadados['perfil_clusters'], use_container_width=True)

    st.markdown("**Rótulos de negócio:**")
    for cluster_id, rotulo in rotulos.items():
        st.write(f"- Cluster {cluster_id}: {rotulo}")

    df_clusterizado['cluster'] = 'Cluster ' + df_clusterizado['cluster'].astype(str)

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("**Separação dos Clusters (Consumo × Tensão)**")
        fig_cluster = px.scatter(
            df_clusterizado, x='tensao_v', y='consumo_kwh', color='cluster',
            color_discrete_sequence=CORES_CLUSTER, template=TEMPLATE_PADRAO,
            hover_data=['regiao', 'hora'], opacity=0.7,
            labels={'tensao_v': 'Tensão (V)', 'consumo_kwh': 'Consumo (kWh)'},
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

    with col_c2:
        st.markdown("**Perfil Médio por Cluster (variáveis normalizadas)**")
        perfil = metadados['perfil_clusters'][['consumo_kwh', 'tensao_v', 'hora']].copy()
        perfil_normalizado = (perfil - perfil.min()) / (perfil.max() - perfil.min())
        perfil_normalizado.index = 'Cluster ' + perfil_normalizado.index.astype(str)
        perfil_long = perfil_normalizado.reset_index().melt(
            id_vars='cluster', var_name='variavel', value_name='valor_normalizado'
        )
        fig_perfil = px.bar(
            perfil_long, x='variavel', y='valor_normalizado', color='cluster',
            barmode='group', color_discrete_sequence=CORES_CLUSTER, template=TEMPLATE_PADRAO,
            labels={'valor_normalizado': 'Nível (0=mín, 1=máx)', 'variavel': 'Variável'},
        )
        st.plotly_chart(fig_perfil, use_container_width=True)

# --- CLIMA ---
with tab_clima:
    st.subheader("Correlação Clima × Consumo")
    st.caption("Dados obtidos via API Open-Meteo (gratuita, sem chave de acesso).")
    if st.button("🔄 Buscar dados climáticos"):
        with st.spinner("Consultando API de clima..."):
            df_com_clima = enriquecer_com_clima(df_filtrado)
        if df_com_clima['temperatura_c'].notna().sum() > 2:
            corr = df_com_clima[['consumo_kwh', 'temperatura_c']].corr().iloc[0, 1]
            st.metric("Correlação Consumo × Temperatura", f"{corr:.3f}")
            fig_clima = px.scatter(
                df_com_clima, x='temperatura_c', y='consumo_kwh', color='regiao', trendline='ols',
                color_discrete_map=CORES_REGIAO, template=TEMPLATE_PADRAO,
                title='Consumo de Energia × Temperatura',
                labels={'temperatura_c': 'Temperatura (°C)', 'consumo_kwh': 'Consumo (kWh)'},
            )
            st.plotly_chart(fig_clima, use_container_width=True)
        else:
            st.warning("Não foi possível obter dados climáticos para o período (verifique conexão ou datas do dataset).")

# --- TEMPO REAL ---
with tab_tempo_real:
    st.subheader("Monitoramento em Tempo Real (Simulação)")
    st.caption("Ao clicar em iniciar, o painel gera leituras fictícias a cada segundo e atualiza o gráfico ao vivo.")

    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        n_passos = st.slider("Duração da simulação (nº de leituras):", 10, 60, 20)
    with col_ctrl2:
        intervalo = st.slider("Intervalo entre leituras (segundos):", 0.5, 3.0, 1.0, 0.5)

    iniciar = st.button("▶️ Iniciar monitoramento ao vivo")

    placeholder_metricas = st.empty()
    placeholder_grafico = st.empty()
    placeholder_tabela = st.empty()

    if iniciar:
        for passo in range(n_passos):
            nova = gerar_nova_leitura(regioes_todas)
            st.session_state.buffer_tempo_real = pd.concat(
                [st.session_state.buffer_tempo_real, pd.DataFrame([nova])], ignore_index=True
            )

            buffer_atual = st.session_state.buffer_tempo_real.tail(30)

            with placeholder_metricas.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Última leitura", f"{nova['consumo_kwh']} kWh", nova['regiao'])
                c2.metric("Tensão", f"{nova['tensao_v']} V")
                c3.metric("Total simulado", len(st.session_state.buffer_tempo_real))

            fig_live = go.Figure()
            for regiao in buffer_atual['regiao'].unique():
                dados_regiao = buffer_atual[buffer_atual['regiao'] == regiao]
                fig_live.add_trace(go.Scatter(
                    x=dados_regiao['timestamp'], y=dados_regiao['consumo_kwh'],
                    mode='lines+markers', name=regiao,
                    line=dict(color=CORES_REGIAO.get(regiao, '#888888')),
                ))
            fig_live.update_layout(
                template=TEMPLATE_PADRAO,
                title=f"Consumo em Tempo Real — leitura {passo + 1}/{n_passos}",
                xaxis_title="Horário", yaxis_title="Consumo (kWh)",
                height=400,
            )
            placeholder_grafico.plotly_chart(fig_live, use_container_width=True, key=f"live_chart_{passo}")

            placeholder_tabela.dataframe(
                buffer_atual.sort_values('timestamp', ascending=False).head(10),
                use_container_width=True,
            )

            time.sleep(intervalo)

        st.success(f"Simulação concluída — {n_passos} leituras geradas.")

    elif not st.session_state.buffer_tempo_real.empty:
        st.info(f"Mostrando as {min(30, len(st.session_state.buffer_tempo_real))} leituras mais recentes da última simulação.")
        buffer_atual = st.session_state.buffer_tempo_real.tail(30)
        fig_live = px.line(
            buffer_atual, x='timestamp', y='consumo_kwh', color='regiao', markers=True,
            color_discrete_map=CORES_REGIAO, template=TEMPLATE_PADRAO,
            title='Última Simulação Registrada',
        )
        st.plotly_chart(fig_live, use_container_width=True)
        st.dataframe(buffer_atual.sort_values('timestamp', ascending=False), use_container_width=True)
    else:
        st.write("Nenhuma simulação executada ainda. Clique em 'Iniciar monitoramento ao vivo'.")

    if not st.session_state.buffer_tempo_real.empty:
        if st.button("🗑️ Limpar histórico de simulação"):
            st.session_state.buffer_tempo_real = df_base.iloc[0:0].copy()
            st.rerun()

# --- RECOMENDAÇÕES ---
with tab_recomendacoes:
    st.subheader("Recomendações para a Gestão Pública")
    df_kpis_regiao = calcular_kpis_smart_grid(df_filtrado)
    regiao_maior_consumo = df_kpis_regiao.iloc[0]['Região']
    regiao_maior_anomalia = df_kpis_regiao.sort_values('Taxa de Anomalias (%)', ascending=False).iloc[0]
    regiao_menor_fator_carga = df_kpis_regiao.sort_values('Fator de Carga (Eficiência)').iloc[0]

    st.markdown(f"""
    Com base nos indicadores, testes estatísticos e padrões de mineração de dados calculados:

    - **{regiao_maior_consumo}** apresenta o maior consumo médio de energia, sendo prioritária
      para ações de eficiência energética.
    - **{regiao_maior_anomalia['Região']}** concentra a maior taxa de anomalias
      ({regiao_maior_anomalia['Taxa de Anomalias (%)']}%), indicando necessidade de inspeção
      preventiva de sensores.
    - **{regiao_menor_fator_carga['Região']}** tem o menor fator de carga
      ({regiao_menor_fator_carga['Fator de Carga (Eficiência)']}), sugerindo picos de demanda
      concentrados — candidata a programas de gestão de demanda.
    """)
    st.info(
        "Recomendações geradas automaticamente a partir dos dados filtrados e devem ser "
        "validadas por especialistas do setor elétrico antes de decisões operacionais."
    )