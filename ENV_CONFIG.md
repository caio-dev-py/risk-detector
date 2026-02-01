# Risk Detector AI - Variáveis de Ambiente

## Arquivo `.env`

O arquivo `.env` contém todas as configurações da aplicação. **Nunca commit este arquivo no git** (já está no `.gitignore`).

### Variáveis de Configuração

| Variável | Valor | Descrição |
|----------|-------|-----------|
| `FLASK_ENV` | `development` | Ambiente de execução (development/production) |
| `FLASK_DEBUG` | `True` | Ativar modo debug |
| `SECRET_KEY` | `risk_detector_secret_key_2026_dev_mode_do_not_use_in_production` | Chave secreta para sessões Flask |
| `FLASK_APP` | `run.py` | Arquivo principal da aplicação |
| `SERVER_HOST` | `0.0.0.0` | Host do servidor |
| `SERVER_PORT` | `5000` | Porta do servidor |
| `MAX_CONTENT_LENGTH` | `16777216` | Tamanho máximo de upload (16MB) |
| `MODEL_PATH` | `src/risk_detector_ai/ml_models/risk_behavior_model.pkl` | Caminho do modelo treinado |
| `DATA_PATH` | `src/risk_detector_ai/data_treino/dados_juntos.csv` | Dados de treinamento |
| `LOG_LEVEL` | `INFO` | Nível de logging |

### Como usar

1. As variáveis são carregadas automaticamente ao iniciar a aplicação
2. Para alterar valores, edite o arquivo `.env`
3. A aplicação lerá as variáveis via `python-dotenv`

### Segurança em Produção

⚠️ **IMPORTANTE**: Antes de deployar em produção:
- Altere `SECRET_KEY` para um valor único e seguro
- Altere `FLASK_DEBUG` para `False`
- Altere `FLASK_ENV` para `production`
- Use um `.env` seguro no servidor (nunca commit)
