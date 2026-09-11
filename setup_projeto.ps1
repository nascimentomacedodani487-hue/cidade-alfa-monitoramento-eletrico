# ==============================================================================
# setup_projeto.ps1
# Organiza a estrutura de pastas e arquivos de suporte do Projeto Cidade Alfa
# ==============================================================================

Write-Host ">>> Organizando estrutura do projeto Cidade Alfa..." -ForegroundColor Cyan

# 1. Criar pastas de dados (se não existirem)
$pastas = @("data\raw", "data\processed")
foreach ($pasta in $pastas) {
    if (-not (Test-Path $pasta)) {
        New-Item -ItemType Directory -Path $pasta -Force | Out-Null
        New-Item -ItemType File -Path "$pasta\.gitkeep" -Force | Out-Null
        Write-Host "  [OK] Criada pasta: $pasta"
    } else {
        Write-Host "  [--] Pasta já existe: $pasta"
    }
}

# 2. Criar requirements.txt
$requirementsContent = @"
pandas
numpy
scipy
seaborn
matplotlib
streamlit
duckdb
"@
Set-Content -Path "requirements.txt" -Value $requirementsContent -Encoding UTF8
Write-Host "  [OK] requirements.txt criado."

# 3. Criar README.md
$readmeContent = @"
# ⚡ Projeto Cidade Alfa — Monitoramento do Sistema Elétrico

FIAP | Fase 5 — Data Intelligence & Analytics

## Sobre o projeto

Pipeline de análise de dados para monitoramento inteligente do sistema elétrico
da Cidade Alfa: consumo de energia, detecção de anomalias, testes estatísticos,
consultas SQL analíticas e dashboard interativo de apoio à decisão para a gestão
pública.

## Estrutura

\`\`\`
src/
├── data/
│   ├── raw/              # dados brutos (CSV de entrada)
│   └── processed/        # dados limpos gerados pelo pipeline
├── preprocessing.py       # limpeza e qualidade de dados (Desafio 1)
├── kpis.py                # cálculo de KPIs de energia
├── stats_analysis.py      # testes de hipótese, correlação e previsão (Desafio 2)
├── sql_layer.py           # consultas SQL analíticas (DuckDB)
└── app.py                 # dashboard interativo (Desafio 3, Streamlit)
\`\`\`

## Como rodar

\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Tecnologias

Python, Pandas, NumPy, SciPy, DuckDB, Streamlit, Seaborn, Matplotlib

## Autor

Projeto desenvolvido para a disciplina de Inteligência Analítica, Estatística
e Tomada de Decisão — FIAP.
"@
Set-Content -Path "README.md" -Value $readmeContent -Encoding UTF8
Write-Host "  [OK] README.md criado."

# 4. Criar .gitignore
$gitignoreContent = @"
.venv/
__pycache__/
*.pyc
.idea/
data/raw/*.csv
data/processed/*.csv
!data/raw/.gitkeep
!data/processed/.gitkeep
"@
Set-Content -Path ".gitignore" -Value $gitignoreContent -Encoding UTF8
Write-Host "  [OK] .gitignore criado."

Write-Host "`n>>> Estrutura organizada com sucesso!" -ForegroundColor Green
Write-Host ">>> Próximo passo: pip install -r requirements.txt" -ForegroundColor Yellow