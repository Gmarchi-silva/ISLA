# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 18:33:20 2023

@author: igort
"""

## Pré-Processamento e limpeza dos dados ##

# importar biblioteca pandas
import pandas as pd

# leitura do conjunto de dados
city_Raw = pd.read_csv(r"C:/Users/igort/Documents/Igor/ISLA GAIA/14.Projeto II/Drive/cities.csv")
product_Raw = pd.read_csv(r"C:/Users/igort/Documents/Igor/ISLA GAIA/14.Projeto II/Drive/product.csv")
sales_Raw = pd.read_csv(r"C:/Users/igort/Documents/Igor/ISLA GAIA/14.Projeto II/Drive/sales.csv")

# corrigir os nomes das cidades (substituir '?' por 'i') e colocar em maiusculas
city_Raw['city_code'] = city_Raw['city_code'].str.replace('?', 'i').str.upper()

# agregação das 3 tabelas numa só
merged_data = pd.merge(sales_Raw, city_Raw, on="store_id")
merged_data = pd.merge(merged_data, product_Raw, on="product_id")

# eliminação dos registos com valores em falta na coluna 'sales' pois também estão em falta na 'revenue'
merged_data = merged_data.dropna(subset=['sales'])

# eliminação dos zeros da coluna 'sales' dado que todos apresentam "Nan" ou "0" na coluna "revenue"
merged_data = merged_data.query("sales != 0")

# eliminação dos registos em que a coluna "revenue" é igual a zero e a coluna "price" é nula 
merged_data = merged_data.query("revenue != 0 or price.notnull()")

# cálculo da 'revenue' quando é igual a zero e existe 'price' e 'sales'
merged_data["revenue"].where(merged_data["revenue"] != 0, merged_data["price"] * merged_data["sales"])

# exportação dos dados brutos agregados para uma 'Raw_table'
merged_data.describe()

merged_data.to_csv(r'C:/Users/igort/Documents/Igor/ISLA GAIA/14.Projeto II/Drive/Raw_table.csv')


## Modelo inicial - média ##

df = merged_data
# seleção das colunas a utilizar neste modelo
df = df[['date', 'store_id', 'sales', 'revenue']]

df.head()
df.describe()

# definir o tipo de dados da coluna 'date' como data
df['date'] = pd.to_datetime(df['date'])

# reindexação - a coluna 'date' passa para o índice
df.set_index("date", inplace=True)

df.info()

# agregação dos dados por semana
df_semana = df.resample('W').sum()

df_semana.info()

df_semana.head()

df_semana.plot()
df_semana.describe()

# criação da variável com os dados para treino de 2017-01-01 a 2019-09-30
df_semana_treino = df_semana.loc["2017-01-01" : "2019-09-30"]

# definir o período de previsão de 2019-07-01 a 2019-09-30
pprevisao = len(df_semana.loc["2019-07-01" : "2019-09-30"])

# Cálculo das previsões com base na média histórica
previsoes_sales = []

for i in range(pprevisao):
    previsao_sales = df_semana_treino['sales'].iloc[ : i + 1].mean()
    previsoes_sales.append(previsao_sales);
   
df_semana_treino['previsoes_sales'] = None

df_semana_treino['previsoes_sales'].iloc[-len(previsoes_sales):] = previsoes_sales


previsoes_revenue = []

for i in range(pprevisao):
    previsao_revenue = df_semana_treino['revenue'].iloc[ : i + 1].mean()
    previsoes_revenue.append(previsao_revenue);
    
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

r2_sales
mae_sales
rmse_sales

real_rev = df_semana_treino.loc['2019-07-07':'2019-09-29', 'revenue']
prev_rev = df_semana_treino.loc['2019-07-07':'2019-09-29', 'previsoes_revenue']

r2_rev = r2_score(real_rev, prev_rev)
mae_rev = mean_absolute_error(real_rev, prev_rev)
rmse_rev = np.sqrt(mean_squared_error(real_rev, prev_rev))

r2_rev
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
