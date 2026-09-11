# ⚡ Projeto Cidade Alfa — Monitoramento do Sistema Elétrico

FIAP | Fase 5 — Data Intelligence & Analytics
Disciplina: Inteligência Analítica, Estatística e Tomada de Decisão
Período: 03/08/2026 a 15/09/2026

## Sobre o projeto

Pipeline de análise de dados para monitoramento inteligente do sistema elétrico da Cidade Alfa: consumo de energia, detecção de anomalias, testes estatísticos, mineração de dados, consultas SQL analíticas, integração com API climática e dashboard interativo de apoio à decisão para a gestão pública.

## Estrutura do Projeto

```text
cidade_alfa_projeto/
├── src/
│   ├── data/
│   │   ├── raw/                 # Dados brutos (CSV de entrada, opcional)
│   │   └── processed/           # Dados limpos gerados pelo pipeline
│   ├── __init__.py
│   ├── app.py                   # Dashboard interativo (Desafio 3 - Streamlit + Plotly)
│   ├── preprocessing.py         # Limpeza e qualidade de dados (Desafio 1)
│   ├── kpis.py                  # Cálculo de KPIs de energia por região
│   ├── stats_analysis.py        # Testes de hipótese, correlação e previsão (Desafio 2)
│   ├── sql_layer.py             # Consultas SQL analíticas (DuckDB)
│   ├── clustering.py            # Mineração de dados — segmentação via K-Means
│   └── api_integration.py       # Integração com API externa (clima - Open-Meteo)
├── README.md
├── requirements.txt
└── .gitignore
```

## Como rodar o projeto

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Execute o dashboard (a partir da pasta `src/`):
```bash
streamlit run app.py
```

> **Atenção:** o Streamlit precisa ser executado via `streamlit run`, não pelo botão "Run" padrão do PyCharm.

## Funcionalidades do Dashboard

- **KPIs** — indicadores de consumo, fator de carga, taxa de anomalias e falhas por região
- **Visualização** — gráficos de consumo por região, tendência temporal e heatmap de anomalias
- **Testes Estatísticos** — ANOVA (diferença de consumo entre regiões) e matriz de correlação
- **Previsão** — projeção de consumo futuro via regressão linear
- **SQL Analítico** — consultas analíticas via DuckDB (agregações, window functions)
- **Mineração de Dados** — segmentação de perfis de consumo via K-Means
- **Clima** — correlação entre temperatura e consumo (API Open-Meteo)
- **Tempo Real** — simulação de monitoramento contínuo com atualização ao vivo
- **Recomendações** — insights automáticos para apoio à decisão da gestão pública

## Tecnologias utilizadas

Python, Pandas, NumPy, SciPy, scikit-learn, DuckDB, Streamlit, Plotly, statsmodels, requests

## Dados

Os dados utilizados são sintéticos, gerados com padrões estatísticos realistas (seed fixo para reprodutibilidade), simulando sensores de consumo elétrico em 5 regiões da Cidade Alfa ao longo de 7 dias.

## Autores

André Bezerra da Costa — RA 573204
Daniel Nascimento de Macedo — RA 572497
Emanuela da Silva Vieira — RA 569139
Guilherme Daniel Laurenti — RA 571240
Manuela Batista de Souza — RA 573642
