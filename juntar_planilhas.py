import pandas as pd



df_bets = pd.read_csv("src/risk_detector_ai/data/data_treino/bets.csv")
df_markets = pd.read_csv("src/risk_detector_ai/data/data_treino/markets.csv")

df_dados_juntos = pd.merge(df_bets, df_markets, left_on="ext_bet_transaction_id", right_on="ext_selection_id" ,how="inner")

df_jogadores = pd.read_csv("src/risk_detector_ai/data/data_treino/jogadores.csv")

df_juncao = pd.merge(df_jogadores, df_dados_juntos, left_on="ID Jogador", right_on="user_id" ,how="inner")


df_juncao.to_csv("src/risk_detector_ai/data/data_treino/dados_completos.csv")