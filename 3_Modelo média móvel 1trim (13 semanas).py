# importar biblioteca pandas
import pandas as pd

# leitura do conjunto de dados - Raw_table
df = pd.read_csv(r"C:/Users/igort/Documents/Igor/ISLA GAIA/14.Projeto II/Drive/Raw_table.csv")


## Modelo inicial - média móvel 1 trimestre (13 semanas) ##

# seleção das colunas a utilizar neste modelo
df = df[['date', 'store_id', 'sales', 'revenue']]

# definir o tipo de dados da coluna 'date' como data
df['date'] = pd.to_datetime(df['date'])

# reindexação - a coluna 'date' passa para o índice
df.set_index("date", inplace=True)

# agregação dos dados por semana
df_semana = df.resample('W').sum()


# Criação da variável com os dados para treino de 2017-01-01 a 2019-09-30
df_semana_treino = df_semana.loc["2017-01-01" : "2019-09-30"]

# Definir o período de previsão de 2019-07-01 a 2019-09-30
pprevisao = len(df_semana.loc["2019-07-01" : "2019-09-30"])

# Cálculo das previsões com base na média histórica das 13 semanas mais recentes
previsoes_sales = []
previsoes_revenue = []

for i in range(pprevisao):
    inicio = i - 13 if i >= 13 else 0
    previsao_sales = df_semana_treino['sales'].iloc[inicio : i + 1].mean()
    previsoes_sales.append(previsao_sales)

    previsao_revenue = df_semana_treino['revenue'].iloc[inicio : i + 1].mean()
    previsoes_revenue.append(previsao_revenue)

df_semana_treino['previsoes_sales'] = None
df_semana_treino['previsoes_sales'].iloc[-len(previsoes_sales):] = previsoes_sales

df_semana_treino['previsoes_revenue'] = None
df_semana_treino['previsoes_revenue'].iloc[-len(previsoes_revenue):] = previsoes_revenue

df_semana_treino


df_semana_treino.plot()

# Avaliação do modelo
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import statsmodels.api as sm

real_sales = df_semana_treino.loc['2019-07-07':'2019-09-29', 'sales']
prev_sales = df_semana_treino.loc['2019-07-07':'2019-09-29', 'previsoes_sales']

r2_sales = r2_score(real_sales, prev_sales)
mae_sales = mean_absolute_error(real_sales, prev_sales)
rmse_sales = np.sqrt(mean_squared_error(real_sales, prev_sales))

r2_sales	# quanto mais próximo de 1, melhor é o resultado
# Um valor negativo indica que o modelo de previsão está pior do que apenas usar a média dos valores reais.
# Pode ser uma indicação que não será um modelo adequado.
mae_sales	# quanto menor, melhor é a precisão
rmse_sales	# quanto menor, melhor é a precisão

real_rev = df_semana_treino.loc['2019-07-07':'2019-09-29', 'revenue']
prev_rev = df_semana_treino.loc['2019-07-07':'2019-09-29', 'previsoes_revenue']

r2_rev = r2_score(real_rev, prev_rev)
mae_rev = mean_absolute_error(real_rev, prev_rev)
rmse_rev = np.sqrt(mean_squared_error(real_rev, prev_rev))

r2_rev
# Um valor negativo indica que o modelo de previsão está pior do que apenas usar a média dos valores reais.
# Pode ser uma indicação que não será um modelo adequado.
mae_rev
rmse_rev

n = len(real_sales)

k = 13  # Número de parâmetros no modelo


sse_sales = ((real_sales - prev_sales) ** 2).sum() # Calcula a soma dos resíduos quadrados (SSE)

aic_sales = n * np.log(sse_sales / n) + 2 * k

bic_sales = n * np.log(sse_sales / n) + k * np.log(n)

print("AIC sales:", aic_sales)
print("BIC sales:", bic_sales)

sse_revenue = ((real_rev - prev_rev) ** 2).sum() # Calcula a soma dos resíduos quadrados (SSE)

aic_revenue = n * np.log(sse_revenue / n) + 2 * k

bic_revenue = n * np.log(sse_revenue / n) + k * np.log(n)

print("AIC revenue:", aic_revenue)
print("BIC revenue:", bic_revenue)
