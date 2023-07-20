# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 01:23:13 2023

@author: jpaul
"""

import numpy as np
import pandas as pd
#import pandas_profiling
import pmdarima as pm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
import re
import openpyxl
from datetime import datetime


def clean_city_code(city_code):
    return city_code.replace('?', 'i').upper()

def fill_missing_values(df):
    fill_values = {'promo_bin_1': 'NA', 'promo_bin_2': 'NA', 'promo_discount_2': 0, 'promo_discount_type_2': 'NA'}
    return df.fillna(fill_values)

def drop_columns(df, columns):
    return df.drop(columns, axis=1, errors='ignore')

def calculate_continuous_week_number(df):
    df['date'] = pd.to_datetime(df['date'])
    df['week_number'] = df['date'].dt.isocalendar().week
    df['year'] = df['date'].dt.year
    min_year = df['year'].min()
    df['continuous_week_number'] = (df['year'] - min_year) * 52 + df['week_number']
    return df



def calculate_continuous_month_number(df):
    df['date'] = pd.to_datetime(df['date'])
    df['month_number'] = df['date'].dt.month
    min_year = df['year'].min()
    df['continuous_month_number'] = (df['year'] - min_year) * 12 + df['month_number']
    return df

path1="C:/Users/jpaul/OneDrive/Ambiente de Trabalho/Projeto2/"
path2="C:/Users/jpaul/OneDrive/Ambiente de Trabalho/Projeto2/Data_Preparation/"

#%%2 Importar ficheiro CSV



filepath = r'C:/Users/jpaul/OneDrive/Ambiente de Trabalho/Projeto2/sales.csv'

sales_raw = pd.read_csv(filepath, dtype={'promo_bin_2': str, 'promo_discount_type_2': str})

filepath1 = r'C:/Users/jpaul/OneDrive/Ambiente de Trabalho/Projeto2/product.csv'

product_raw = pd.read_csv(filepath1, sep = ';')

filepath2 = r'C:/Users/jpaul/OneDrive/Ambiente de Trabalho/Projeto2/cities.csv'

city_raw = pd.read_csv(filepath2, sep = ';')

#%% Análise exploratória de dados foi feita em R e em python



#%%Tratamento de dados

# Clean city code
city_raw['city_code'] = city_raw['city_code'].apply(clean_city_code)

# Fill missing values
sales_raw = fill_missing_values(sales_raw)

# Merge data
merged_data = pd.merge(sales_raw, city_raw, on="store_id")
merged_data = pd.merge(merged_data, product_raw, on="product_id")

# Drop columns
columns_to_drop = ['city_id_old', 'Unnamed: 0', 'country_id', 'hierarchy1_id', 'hierarchy2_id',
                   'hierarchy3_id', 'hierarchy4_id', 'hierarchy5_id', 'product_length', 'product_depth', 'product_width', 'cluster_id']
merged_data = drop_columns(merged_data, columns_to_drop)

# Drop rows with null sales
merged_data = merged_data.dropna(subset=['sales'])

# Calculate continuous week number
merged_data = calculate_continuous_week_number(merged_data)

#contar os NaN
count_nan = merged_data.isna().sum()


# Valor para filtrar
valor_filtro = 144

# Remover linhas com a semana 144 porque só tem valor de 1 dia
merged_data = merged_data.loc[merged_data['continuous_week_number'] != valor_filtro]




#%%modelagem para SARIMAX variáveis exógenas

# Somar as colunas especificadas por linha
merged_data['stock_inicial'] = merged_data['sales'] + merged_data['stock']
merged_data=calculate_continuous_month_number(merged_data)

#Lista do tipo de promoções associadas á promo_bin_1
#valores_unicos_Promobin_1 = merged_data['promo_bin_1'].unique().tolist()

#Tipo de promoções promo_bin_1 por dia
merged_data['Probin1_very_low']= np.where((merged_data['promo_bin_1'] == 'verylow'), 1, 0)
merged_data['Probin1_low']= np.where((merged_data['promo_bin_1'] == 'low'), 1, 0)
merged_data['Probin1_moderate']= np.where((merged_data['promo_bin_1'] == 'moderate'), 1, 0)
merged_data['Probin1_high']= np.where((merged_data['promo_bin_1'] == 'high'), 1, 0)
merged_data['Probin1_very_high']= np.where((merged_data['promo_bin_1'] == 'veryhigh'), 1, 0)

#Lista do tipo de promoções associadas á promo_bin_1
#valores_unicos_Promobin_2 = merged_data['promo_bin_2'].unique().tolist()

#Tipo de promoções promo_bin_2 por dia
merged_data['Probin2_very_low']= np.where((merged_data['promo_bin_2'] == 'verylow'), 1, 0)
merged_data['Probin2_high']= np.where((merged_data['promo_bin_2'] == 'high'), 1, 0)
merged_data['Probin2_very_high']= np.where((merged_data['promo_bin_2'] == 'veryhigh'), 1, 0)

#Lista do tipo de promoções associadas á promo_bin_1
valores_unicos_Promodiscount2 = merged_data['promo_discount_2'].unique().tolist()

merged_data['promo_discount_16']= np.where((merged_data['promo_discount_2'] == 16), 1, 0)
merged_data['promo_discount_20']= np.where((merged_data['promo_discount_2'] == 20), 1, 0)
merged_data['promo_discount_35']= np.where((merged_data['promo_discount_2'] == 35), 1, 0)
merged_data['promo_discount_40']= np.where((merged_data['promo_discount_2'] == 40), 1, 0)
merged_data['promo_discount_50']= np.where((merged_data['promo_discount_2'] == 50), 1, 0)

#Primeiro fizemos um groupby date pq queremos o total de stock_inicial por dia
merged_data= merged_data.groupby(['store_id', 'date','storetype_id','city_code']).agg({
    'sales': 'sum',
    'revenue': 'sum',
    'stock_inicial': 'sum',
    'continuous_week_number':'mean',
    'week_number':'mean',
    'store_size':'mean',
    'Probin1_very_low':'sum',
    'Probin1_low':'sum',
    'Probin1_moderate':'sum',
    'Probin1_high':'sum',
    'Probin1_very_high':'sum',
    'Probin2_very_low':'sum',
    'Probin2_high':'sum',
    'Probin2_very_high':'sum',
    'promo_discount_16':'sum',
    'promo_discount_20':'sum',
    'promo_discount_35':'sum',
    'promo_discount_40':'sum',
    'promo_discount_50':'sum',
}).reset_index()

#Neste segundo Groupby fizemos por continuous_week_number e neste caso escolhemos o stock inicial máximo dos 7 dias que representam a semana
merged_data= merged_data.groupby(['store_id', 'continuous_week_number','storetype_id','city_code']).agg({
    'sales': 'sum',
    'revenue': 'sum',
    'stock_inicial': 'max',
    'week_number':'mean',
    'store_size':'mean',
    'Probin1_very_low':'sum',
    'Probin1_low':'sum',
    'Probin1_moderate':'sum',
    'Probin1_high':'sum',
    'Probin1_very_high':'sum',
    'Probin2_very_low':'sum',
    'Probin2_high':'sum',
    'Probin2_very_high':'sum',
    'promo_discount_16':'sum',
    'promo_discount_20':'sum',
    'promo_discount_35':'sum',
    'promo_discount_40':'sum',
    'promo_discount_50':'sum',    
}).reset_index()

# Definir o intervalo de valores
valor_min = 13
valor_max = 25

# Atribuir 1 para valores dentro do intervalo e 0 para valores fora
merged_data['Primavera'] = np.where((merged_data['week_number'] >= valor_min) & (merged_data['week_number'] <= valor_max), 1, 0)

# Definir o intervalo de valores
valor_min = 26
valor_max = 39

# Atribuir 1 para valores dentro do intervalo e 0 para valores fora
merged_data['Verao'] = np.where((merged_data['week_number'] >= valor_min) & (merged_data['week_number'] <= valor_max), 1, 0)

# Definir o intervalo de valores
valor_min = 40
valor_max = 51

# Atribuir 1 para valores dentro do intervalo e 0 para valores fora
merged_data['Outono'] = np.where((merged_data['week_number'] >= valor_min) & (merged_data['week_number'] <= valor_max), 1, 0)

#Definir as semanas com feriados
feriado_semanas=[1, 16, 17, 18, 20, 25, 26, 28, 29, 35, 36, 43, 44]

merged_data['Feriados']=merged_data['week_number'].isin(feriado_semanas).astype(int)


#df.to_csv("".join([path,'Tabela_variaveis_exogenas.csv']),sep=',',index=False)


#%%Lojas com menos de 143 semanas no total
#Como temos lojas com valores de semana muito abaixo de 143 decidimos fazer uma análise particular

Lojas_normais=merged_data.groupby('store_id').filter(lambda x: len(x) == 143)
contagem_normais = Lojas_normais['store_id'].value_counts()
Lojas_anormais_total=merged_data.groupby('store_id').filter(lambda x: len(x) < 143)
contagem_anormais = Lojas_anormais_total['store_id'].value_counts()




#%%Grafico para analisar lojas anormais totais
# Obter a lista de store_id presentes no DataFrame
store_ids_anormais_totais = Lojas_anormais_total['store_id'].unique()

# Definir o número de colunas, colunas e o espaçamento entre os subplots
num_cols = 4
spacing = 0.2
num_rows = 3

# Configurar o tamanho da figura e a matriz de subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 10))

# Plotar os gráficos de linhas para cada store_id em subplots separados
for i, store_id in enumerate(store_ids_anormais_totais):
    row = i // num_cols
    col = i % num_cols

    ax = axs[row, col] if num_rows > 1 else axs[col]
    df_filtered = Lojas_anormais_total[Lojas_anormais_total['store_id'] == store_id]
    ax.plot(df_filtered['continuous_week_number'], df_filtered['revenue'])
    ax.set_ylabel('Revenue')
    ax.set_title('Store ID: ' + store_id)

# Remover subplots vazios, se necessário
if len(store_ids_anormais_totais) < num_rows * num_cols:
    for i in range(len(store_ids_anormais_totais), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].axis('off')

# Configurar o rótulo do eixo x para todos os subplots
for ax in axs.flat:
    ax.set_xlabel('Continuous Week Number')

# Ajustar o espaçamento entre os subplots e as margens da figura
fig.tight_layout(pad=spacing)

plt.savefig("".join([path2, 'Lojas anormais totais', store_id,'.png']))
#plt.show()
plt.close(fig)
# Exibir a figura com os subplots
plt.show()


#%%Tratamento das lojas anormais
#%% Remover a store_id S0136 pq é diferente das outras lojas anormais
Lojas_anormais = Lojas_anormais_total.drop(Lojas_anormais_total[Lojas_anormais_total['store_id'] == 'S0136'].index)



#%% Remover das lojas_anormais todos os valores de revenue igual 0 uma vez que representam a abertura da loja, não interessam para a previsão
Lojas_anormais = Lojas_anormais.drop(Lojas_anormais[Lojas_anormais['revenue'] == 0].index)


#%% Remover das lojas_anormais todos os valores que sejam uma abertura para teste da loja
lojas_remover=['S0005', 'S0036', 'S0046', 'S0061', 'S0071', 'S0076', 'S0092', 'S0109']

Lojas_anormais = Lojas_anormais.drop(Lojas_anormais[(Lojas_anormais['store_id'].isin(lojas_remover)) & (Lojas_anormais['continuous_week_number'] == 53)].index)



#%% Gráfico para analisar as revenues das lojas_anormais por semana
#%%Grafico para analisar lojas anormais

store_ids_anormais = Lojas_anormais['store_id'].unique()
# Definir o número de colunas, colunas e o espaçamento entre os subplots
num_cols = 4
spacing = 0.2
num_rows = 3

# Configurar o tamanho da figura e a matriz de subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 10))

# Plotar os gráficos de linhas para cada store_id em subplots separados
for i, store_id in enumerate(store_ids_anormais):
    row = i // num_cols
    col = i % num_cols

    ax = axs[row, col] if num_rows > 1 else axs[col]
    df_filtered = Lojas_anormais[Lojas_anormais['store_id'] == store_id]
    ax.plot(df_filtered['continuous_week_number'], df_filtered['revenue'])
    ax.set_ylabel('Revenue')
    ax.set_title('Store ID: ' + store_id)

# Remover subplots vazios, se necessário
if len(store_ids_anormais) < num_rows * num_cols:
    for i in range(len(store_ids_anormais), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axs[row, col].axis('off')

# Configurar o rótulo do eixo x para todos os subplots
for ax in axs.flat:
    ax.set_xlabel('Continuous Week Number')

# Ajustar o espaçamento entre os subplots e as margens da figura
fig.tight_layout(pad=spacing)

plt.savefig("".join([path2, 'Lojas anormais', store_id,'.png']))

# Exibir a figura com os subplots
plt.show()



#%% Loja S0136 - com sazonalidade não estando aberta todo o ano
Lojas_anormais_S0136 = Lojas_anormais_total[Lojas_anormais_total['store_id'] == 'S0136']
Lojas_anormais_S0136[Lojas_anormais_S0136['revenue'] == 0]

#Gráfico das loja S0136
plt.plot(Lojas_anormais_S0136['continuous_week_number'], Lojas_anormais_S0136['revenue'])
plt.title('S0136')

# Criar e preencher com '0' as linhas intercalares inexistentes (eliminadas inicialmente por terem valores nulos)

all_weeks = pd.DataFrame({'continuous_week_number': range(int(Lojas_anormais_S0136['continuous_week_number'].min()), int(Lojas_anormais_S0136['continuous_week_number'].max()) + 1)})


Lojas_anormais_S0136 = pd.merge(all_weeks, Lojas_anormais_S0136, on='continuous_week_number', how='left')
#Lojas_anormais_S0136 = pd.merge(all_weeks, Lojas_anormais_S0136[['continuous_week_number', 'revenue', 'sales']], on=['continuous_week_number'], how='left')

Lojas_anormais_S0136['store_id'] = Lojas_anormais_S0136['store_id'].fillna('S0136')
#Ciclo para colocar o week_number correto
for item in range(0,116):
    if (Lojas_anormais_S0136['continuous_week_number'][item] > 104):
        Lojas_anormais_S0136['week_number'][item] = Lojas_anormais_S0136['continuous_week_number'][item]-104
    elif (Lojas_anormais_S0136['continuous_week_number'][item] > 52):
       Lojas_anormais_S0136['week_number'] [item]= Lojas_anormais_S0136['continuous_week_number'][item]-52
    else:
        Lojas_anormais_S0136['week_number'] [item]= Lojas_anormais_S0136['continuous_week_number'][item]

Lojas_anormais_S0136 = Lojas_anormais_S0136.fillna(0)

# Visualização gráfica
plt.plot(Lojas_anormais_S0136['continuous_week_number'], Lojas_anormais_S0136['revenue'])
plt.title('S0136')

''' Pela análise efetuada a loja terá fechado na semana 114 como habitualmente fez nos períodos homólogos
de 2017 e 2018, pelo que a previsão para outubro de 2019 é que esteja fechada e por isso não haja vendas.
No entanto vamos introduzir no modelo SARIMA e verificar os resultados/previsões'''

#%% Modelagem



#%% Preparação


# Agrupar o merged_data que tem as lojas normais com lojas_anormais e com a loja S0136
df_final = Lojas_normais
df_final = df_final.append(Lojas_anormais)
df_final = df_final.append(Lojas_anormais_S0136)

contagem_final = df_final['store_id'].value_counts()

df_final.to_csv("".join([path1,'df_final.csv']),sep=',',index=False)

