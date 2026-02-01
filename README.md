# Risk Detector AI

Sistema de detecção de jogadores de risco em plataformas de apostas esportivas utilizando **Isolation Forest** (aprendizado não-supervisionado).

## 🎯 Objetivo

Identificar jogadores com comportamentos anômalos e potencialmente arriscados por meio de agregação de métricas de apostas e análise de padrões via modelo de machine learning.

## 📊 Métricas Analisadas

Cada jogador é caracterizado por:
- **Contagem de Apostas** (`bet_count`) — frequência de engajamento
- **Valor Total Apostado** (`total_stake`) — volume financeiro investido
- **Valor Médio por Aposta** (`avg_stake`) — padrão de tamanho de aposta
- **Odds Médias** (`avg_odds`) — nível de risco matemático
- **Taxa de Retorno** (`return_ratio`) — performance (lucros/investimento)
- **Dias desde Última Aposta** (`days_since_last_bet`) — recência de atividade

## 🚀 Início Rápido

### Pré-requisitos
- Python 3.10+
- Poetry (gerenciador de dependências)

### Instalação

```bash
# Clone ou navegue até o diretório do projeto
cd "c:\Users\Caio Araujo\Documents\Programação\Programas Esportiva\risk_detector_ai"

# Instale as dependências
poetry install

# Configure as variáveis de ambiente (opcional)
# Crie um arquivo .env na raiz com:
# SERVER_HOST=0.0.0.0
# SERVER_PORT=5000
# FLASK_DEBUG=True
# SECRET_KEY=your-secret-key-here
```

### Treinar o Modelo

```bash
poetry run python -c "from src.risk_detector_ai.train import train_isolation_forest; train_isolation_forest()"
```

Isso treina o modelo em `src/risk_detector_ai/data/data_treino/dados_completos.csv` e salva em `src/risk_detector_ai/ml_models/risk_model.pkl`.

### Rodar o Servidor

```bash
poetry run python run.py
```

O servidor estará disponível em `http://127.0.0.1:5000`.

## 📁 Estrutura do Projeto

```
risk_detector_ai/
├── src/
│   └── risk_detector_ai/
│       ├── app.py                 # Flask app factory + rotas (treino, upload, visualização)
│       ├── train.py               # Isolation Forest training logic
│       ├── features.py            # Feature aggregation por jogador
│       ├── models.py              # SQLAlchemy ORM models (se aplicável)
│       ├── config.py              # Configurações
│       ├── __init__.py
│       ├── data/
│       │   └── data_treino/
│       │       └── dados_completos.csv   # Training data (risk players)
│       ├── ml_models/             # Artefatos treinados (.pkl)
│       ├── instance/              # Database (risk.db)
│       ├── uploads/               # Arquivos CSV enviados + resultados
│       ├── templates/
│       │   ├── base.html          # Layout base
│       │   ├── index.html         # Dashboard principal
│       │   ├── analyses.html      # Listagem de análises
│       │   └── analysis_view.html # Visualização de uma análise
│       └── static/
│           └── css/
│               ├── dark_theme.css
│               ├── reset.css
│               └── style.css
├── data_teste/                    # Dados de teste (CSV de entrada)
├── pyproject.toml                 # Dependências (Poetry)
├── run.py                         # Entry point
├── .env                           # Variáveis de ambiente (não commitar)
├── .gitignore
└── README.md
```

## 🔗 Rotas Disponíveis

| Rota | Método | Descrição |
|------|--------|-----------|
| `/` | GET | Dashboard principal (resumo e upload) |
| `/upload` | POST | Upload de CSV para análise |
| `/analyses` | GET | Lista todas as análises salvas |
| `/view` | GET | Visualiza uma análise específica (query param: `filename`) |
| `/uploads/<filename>` | GET | Baixa arquivo CSV |

## 📋 Fluxo de Uso

1. **Upload**: Envie um CSV com coluna `ID Jogador` e colunas de apostas (stake, odds, etc).
2. **Processamento**: O sistema agrupa dados por jogador e computa agregações.
3. **Predição**: Usa o modelo Isolation Forest para calcular anomalia (risk score 0-1).
4. **Explicação**: Gera razões rule-based (ex.: "Avg stake muito alto", "Perdas sistemáticas").
5. **Visualização**: Exibe tabela ordenada por risco (maior → menor) com razões.

## 🤖 Modelo de Machine Learning

- **Algoritmo**: Isolation Forest (n_estimators=300, contamination=0.01)
- **Treinamento**: Não-supervisionado (sem rótulos)
- **Preprocessamento**: StandardScaler + Imputação (mediana)
- **Output**: Risk Score normalizado (0.0 a 1.0)

## 📊 Exemplo de Resultado

Após upload de CSV, o sistema retorna:

```
ID Jogador | Risk Score | Avg Stake | Win Rate | Motivos
-----------|------------|-----------|----------|------------------
12345      | 0.892      | 500.50    | -0.15    | Avg stake muito alto, Perdas sistemáticas
67890      | 0.654      | 250.00    | 0.85     | Alto número de apostas
```

## 🛠️ Desenvolvimento

### Instalar pacotes adicionais

```bash
poetry add <package-name>
```

### Rodar testes (se implementado)

```bash
poetry run pytest
```

### Debugger PIN

Quando em modo debug, procure pelo PIN do Werkzeug nos logs da aplicação para acessar o console remoto.

## 📝 Variáveis de Ambiente (.env)

```env
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=dev-secret-key-change-in-prod
SERVER_HOST=0.0.0.0
SERVER_PORT=5000
MAX_CONTENT_LENGTH=16777216
UPLOAD_FOLDER=src/risk_detector_ai/uploads/test_files
MODEL_PATH=src/risk_detector_ai/ml_models/risk_model.pkl
```

## ⚠️ Notas Importantes

- **Dados Sensíveis**: Não commite arquivos `.env`, `*.pkl`, `*.db` ou pasta `uploads/`.
- **Banco de Dados**: O SQLite é criado automaticamente na pasta `instance/`.
- **Modelo**: Retreine regularmente com novos dados para manter assertividade.
- **Performance**: Contamination=0.01 (1%) significa ~1% dos usuários será flagged como risco.

## 📞 Suporte

Para dúvidas ou issues:
1. Verifique os logs do servidor (stdout).
2. Confirme que o arquivo CSV tem as colunas esperadas.
3. Valide o formato de dados (sem quebras de padrão).

---

**Última atualização**: Fevereiro 2026
