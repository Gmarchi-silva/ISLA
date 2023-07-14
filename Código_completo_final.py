#% Projeto 2

#%%1 Importar as bibliotecas necessárias

import numpy as np
import pandas as pd
#import pandas_profiling
import pmdarima as pm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import math
import re
import openpyxl
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

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

def evaluate_arima_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    summary = pd.DataFrame({'MAE': [mae], 'MSE': [mse], 'RMSE': [rmse], 'MAPE': [mape]})
    return summary

def calculate_continuous_month_number(df):
    df['date'] = pd.to_datetime(df['date'])
    df['month_number'] = df['date'].dt.month
    min_year = df['year'].min()
    df['continuous_month_number'] = (df['year'] - min_year) * 12 + df['month_number']
    return df

path1="C:/Users/jpaul/OneDrive/Ambiente de Trabalho/Projeto2/"

path2="C:/Users/jpaul/OneDrive/Ambiente de Trabalho/Projeto2/SARIMA_gráficos_revenues/"

path3="C:/Users/jpaul/OneDrive/Ambiente de Trabalho/Projeto2/Sarimax_revenues/"

path4="C:/Users/jpaul/OneDrive/Ambiente de Trabalho/Projeto2/Sarimax_cluster/"

path5= "C:/Users/jpaul/OneDrive/Ambiente de Trabalho/Projeto2/Sarimax_lojas_STO3_03_indv/"
#%%2 Importar ficheiro CSV

filepath =  r'C:/Users/jpaul/OneDrive/Ambiente de Trabalho/Projeto2/Tabela_geral.csv'

df = pd.read_csv(filepath)

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

plt.savefig("".join([path4, 'Lojas anormais', store_id,'.png']))
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

df_final.to_csv(r'C:/Users/igort/Documents/Igor/ISLA GAIA/14.Projeto II/Drive/grouped2_df.csv')



#%% SARIMA e ARIMA (com auto_arima)

#%Agrupar os dados
grouped_df= df_final.groupby(['store_id', 'continuous_week_number','storetype_id','city_code']).agg({
    'sales': 'sum',
    'revenue': 'sum',
    'store_size': 'mean'
    
}).reset_index()


df_results_63 = pd.DataFrame()
valores_unicos = grouped_df['store_id'].unique().tolist()


for store_id in valores_unicos:

    
    #store_id='S0002'
    
    
    # Filtrar os dados para a loja específica
    store_data = grouped_df[grouped_df['store_id'] == store_id]
     
    tamanho = len(store_data['store_id'])
    if tamanho > 79:
        
        
        inicio=store_data.index[0]
        # Definir o número da semana para o início e o fim do conjunto de treino e teste
        train_start_week = inicio # Definir o número da semana de início do treino
        train_end_week = inicio+len(store_data)-10 # Definir o número da semana de término do treino
        test_start_week = inicio+len(store_data)-9  # Definir o número da semana de início do teste
        test_end_week = inicio+len(store_data)-1   # Definir o número da semana de término do teste
        
        # Filtrar os dados para os períodos de treino e teste
        
        train_data = store_data.loc[train_start_week:train_end_week, 'revenue']
        time_train_data = store_data.loc[train_start_week:train_end_week,'continuous_week_number']
        test_data = store_data.loc[test_start_week:test_end_week, 'revenue']
        time_test_data = store_data.loc[test_start_week:test_end_week,'continuous_week_number']
            
        # Ajustar o modelo SARIMA
        modelo_arima = pm.auto_arima(train_data, seasonal=True, trace=True, m=52)
        modelo=modelo_arima.fit(train_data)
           
            
        # Fazer previsões no conjunto de teste
        
        forecast = modelo_arima.predict(n_periods=len(test_data))
        coeficientes = modelo_arima.params()
            
        #Indice do forecast direito
        forecast.index=test_data.index
        # Calcular os resíduos
        erros_estimacao = test_data - forecast
        residuos=modelo_arima.resid()
        modelo_arima_summary = modelo_arima.summary()
            
        # Exibir as previsões para a loja
        print(f"Previsões para a loja {store_id}:\n{forecast}\n")
        #Armazenar previsões
        df_forecast[store_id] = forecast
        
        #coeficientes
        print(coeficientes)
            
        #summary=evaluate_arima_model(test_data, forecast)
        print(modelo_arima_summary)
            
        df_results=pd.DataFrame()
        # Criar um DataFrame com os resultados dos coeficientes e erros padrão
        df_results = pd.DataFrame({'Tipo de Coeficiente':coeficientes.index,'Coeficiente': coeficientes,'Erro Padrão': modelo_arima.bse()})
        
        # Adicionar os valores de AIC e BIC aos resultados
        df_results['AIC'] = modelo_arima.aic()
        df_results['BIC'] = modelo_arima.bic()
        df_results['store_id']=store_id
        df_results['Modelo'] = modelo
        store_data.reset_index(drop=True, inplace=True)
        df_results['store_type']=store_data['storetype_id'].loc[1]
        df_results['store_size']=store_data['store_size'].loc[1]
        df_results['city_code']= store_data['city_code'].loc[1]
        
        df_results_63=pd.concat([df_results_63,df_results],axis=0)
        
            
            
        #Plot Resíduos
            
        fig = plt.figure()
        modelo_arima.plot_diagnostics(figsize=(10,8))
        #plt.savefig("".join([path2, 'Resíduos', store_id,'.png']))
        #plt.show()
        plt.close(fig)
            
        # Plotar os dados dos erros de estimação
        fig = plt.figure()
        plt.figure(figsize=(15, 12))
        plt.scatter(time_test_data, erros_estimacao)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Semanas',fontsize=(20))
        plt.ylabel('Erros de Estimação',fontsize=(20))
        plt.title('Gráfico de Erros de estimação do Modelo SARIMA',fontsize=(30))
        #plt.savefig("".join([path2, 'Erros', store_id,'.png']))
        #plt.show()
        plt.close(fig)
            
        # Plotar os dados originais, as previsões e os dados de teste
        fig = plt.figure()
        plt.figure(figsize=(20, 15))
        plt.plot(store_data['continuous_week_number'], store_data['revenue'], label='Dados Originais')
        plt.plot(time_test_data, forecast, label='Previsões')
        plt.plot(time_test_data, test_data, label='Dados de Teste')
        plt.xlabel('Semanas',fontsize=(20))
        plt.ylabel('Revenue',fontsize=(20))
        plt.title('Previsões SARIMA',fontsize=(30))
        plt.legend()
        #plt.savefig("".join([path2, 'Previsão', store_id,'.png']))
        #plt.show()
        plt.close(fig)
        
    else:
        inicio=store_data.index[0]
        # Definir o número da semana para o início e o fim do conjunto de treino e teste
        train_start_week = inicio # Definir o número da semana de início do treino
        train_end_week = inicio+len(store_data)-10 # Definir o número da semana de término do treino
        test_start_week = inicio+len(store_data)-9  # Definir o número da semana de início do teste
        test_end_week = inicio+len(store_data)-1   # Definir o número da semana de término do teste
        
        # Filtrar os dados para os períodos de treino e teste
        
        train_data = store_data.loc[train_start_week:train_end_week, 'revenue']
        time_train_data = store_data.loc[train_start_week:train_end_week,'continuous_week_number']
        test_data = store_data.loc[test_start_week:test_end_week, 'revenue']
        time_test_data = store_data.loc[test_start_week:test_end_week,'continuous_week_number']
            
        # Ajustar o modelo SARIMA
        modelo_arima = pm.auto_arima(train_data, seasonal=False, trace=True)
        modelo=modelo_arima.fit(train_data)
           
            
        # Fazer previsões no conjunto de teste
        
        forecast = modelo_arima.predict(n_periods=len(test_data))
        coeficientes = modelo_arima.params()
            
        #Indice do forecast direito
        forecast.index=test_data.index
        # Calcular os resíduos
        erros_estimacao = test_data - forecast
        residuos=modelo_arima.resid()
        modelo_arima_summary = modelo_arima.summary()
            
        # Exibir as previsões para a loja
        print(f"Previsões para a loja {store_id}:\n{forecast}\n")
            
        #coeficientes
        print(coeficientes)
            
        #summary=evaluate_arima_model(test_data, forecast)
        print(modelo_arima_summary)
            
        df_results=pd.DataFrame()
        # Criar um DataFrame com os resultados dos coeficientes e erros padrão
        df_results = pd.DataFrame({'Tipo de Coeficiente':coeficientes.index,'Coeficiente': coeficientes,'Erro Padrão': modelo_arima.bse()})
        
        # Adicionar os valores de AIC e BIC aos resultados
        df_results['AIC'] = modelo_arima.aic()
        df_results['BIC'] = modelo_arima.bic()
        df_results['store_id']=store_id
        df_results['Modelo'] = modelo
        store_data.reset_index(drop=True, inplace=True)
        df_results['store_type']=store_data['storetype_id'].loc[1]
        df_results['store_size']=store_data['store_size'].loc[1]
        df_results['city_code']= store_data['city_code'].loc[1]
        
        df_results_63=pd.concat([df_results_63,df_results],axis=0)
        
            
            
        #Plot Resíduos
            
        fig = plt.figure()
        modelo_arima.plot_diagnostics(figsize=(10,8))
        #plt.savefig("".join([path2, 'Resíduos', store_id,'.png']))
        #plt.show()
        plt.close(fig)
            
        # Plotar os dados dos erros de estimação
        fig = plt.figure()
        plt.figure(figsize=(15, 12))
        plt.scatter(time_test_data, erros_estimacao)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Semanas',fontsize=(20))
        plt.ylabel('Erros de Estimação',fontsize=(20))
        plt.title('Gráfico de Erros de estimação do Modelo SARIMA',fontsize=(30))
        #plt.savefig("".join([path2, 'Erros', store_id,'.png']))
        #plt.show()
        plt.close(fig)
            
        # Plotar os dados originais, as previsões e os dados de teste
        fig = plt.figure()
        plt.figure(figsize=(20, 15))
        plt.plot(store_data['continuous_week_number'], store_data['revenue'], label='Dados Originais')
        plt.plot(time_test_data, forecast, label='Previsões')
        plt.plot(time_test_data, test_data, label='Dados de Teste')
        plt.xlabel('Semanas',fontsize=(20))
        plt.ylabel('Revenue',fontsize=(20))
        plt.title('Previsões SARIMA',fontsize=(30))
        plt.legend()
        #plt.savefig("".join([path2, 'Previsão', store_id,'.png']))
        #plt.show()
        plt.close(fig)
    

# Salvar o DataFrame no arquivo Excel
#df_results_63.to_csv("".join([path2,'resultados_Sarima_revenues.csv']),sep=',',index=False)

#%%% SARIMAX


#%%Retirar dados reais do data set
#%Agrupar os dados
grouped_df= df_final.groupby(['store_id', 'continuous_week_number','storetype_id','city_code']).agg({
    'sales': 'sum',
    'revenue': 'sum',
    'store_size': 'mean'
    
}).reset_index()
#%dados reais retirados para comparação com os previstos para o cluster
ST03_03 = ['S0141', 'S0120', 'S0077', 'S0143', 'S0068', 'S0039', 'S0016', 'S0080']
dados_reais = grouped_df[grouped_df['store_id'].isin(T03_03)][['store_id', 'continuous_week_number', 'revenue']].copy()
dados_reais = dados_reais.pivot(index='continuous_week_number', columns='store_id', values='revenue')
dados_reais = dados_reais.reset_index()
dados_reais = dados_reais[dados_reais['continuous_week_number'].between(135, 143)]
#dados_reais.to_csv("".join([path5,'resultados_reais.csv']),sep=',',index=False)

#%%Sarimax para todas as lojas individualmente

#%Data frame grouped com todas as variaveis exogenas abaixo e outras informaçoes importantes
grouped_df= df_final.groupby(['store_id', 'continuous_week_number','storetype_id','city_code']).agg({
    'sales':'sum',
    'revenue': 'sum',
    'stock_inicial': 'max',
    'store_size': 'mean',
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
    'Primavera':'mean',
    'Verao':'mean',
    'Outono':'mean',
    'Feriados':'mean',
}).reset_index()


#grouped2_df.to_csv('Loja_Sales_Revenue2.csv', index=False)
#df_forecast = pd.DataFrame()

valores_unicos = grouped_df['store_id'].unique().tolist()
df_results_63 = pd.DataFrame()


for store_id in valores_unicos:  
    #store_id='S0141'
    
    # Filtrar os dados para a loja específica
    store_data = grouped_df[grouped_df['store_id'] == store_id]
    store_data.index=range(0,len(store_data['revenue']))     
    tamanho = len(store_data['store_id'])
    
    if tamanho > 79:  
        inicio=store_data.index[0]
        
        # Definir o número da semana para o início e o fim do conjunto de treino e teste
        train_start_week = inicio # Definir o número da semana de início do treino
        train_end_week = inicio+len(store_data)-10 # Definir o número da semana de término do treino
        test_start_week = inicio+len(store_data)-9  # Definir o número da semana de início do teste
        test_end_week = inicio+len(store_data)-1   # Definir o número da semana de término do teste
        
        # Filtrar os dados para os períodos de treinamento e teste
        colunas_remover=['sales','continuous_week_number','store_id']
        dados_final=store_data.drop(colunas_remover, axis=1)
        train_data = dados_final.loc[train_start_week:train_end_week]
        time_train_data = store_data.loc[train_start_week:train_end_week,'continuous_week_number']
        test_data = dados_final.loc[test_start_week:test_end_week]
        time_test_data = store_data.loc[test_start_week:test_end_week,'continuous_week_number']
        
        # Ajustar o modelo ARIMA
        Exogenas_remover=['revenue','storetype_id','city_code','store_size']
        #,'promo_discount_16','promo_discount_20','promo_discount_35','promo_discount_40','promo_discount_50','Primavera','Verao','Outono'
        Exogenas=train_data.drop(Exogenas_remover, axis=1)
        #Exogenas=['stock_inicial']
        modelo_arima0 = pm.auto_arima(train_data['revenue'], X=Exogenas, supress_warnings=True,stepwise=True, seasonal=True, trace=True, m=52)
        modelo=modelo_arima0.fit(train_data['revenue'], X=Exogenas)
        # Fazer previsões no conjunto de teste
        print(train_data.dtypes)

        #forecast = modelo_arima.predict(n_periods=len(test_data))
        Exogenas_test=test_data.drop(Exogenas_remover, axis=1)
        forecast = modelo_arima0.predict(n_periods=len(test_data),X=Exogenas_test)
        coeficientes = modelo_arima0.params()
        
        #Indice do forecast direito
        forecast.index=test_data.index
        # Calcular os resíduos
        erros_estimacao = test_data['revenue'] - forecast
        residuos1=modelo_arima0.resid()
        modelo_arima_summary = modelo.summary()
        
        # Exibir as previsões para a loja
        print(f"Previsões para a loja {store_id}:\n{forecast}\n")
        #Armazenar previsões
        df_forecast[store_id] = forecast
        #coeficientes
        print(coeficientes)
        coeficientes_exogenas = modelo_arima0.params()[modelo_arima0.order[2]:]
        print(coeficientes_exogenas)
        #summary=evaluate_arima_model(test_data, forecast)
        print(modelo_arima_summary)
        
        df_results=pd.DataFrame()
        # Criar um DataFrame com os resultados dos coeficientes e erros padrão
        df_results = pd.DataFrame({'Tipo de Coeficiente':coeficientes.index,'Coeficiente': coeficientes,'p_value': modelo_arima0.pvalues(), 'inter_conf_inf':modelo_arima0.conf_int()[0],'inter_conf_sup':modelo_arima0.conf_int()[1]})
    
        # Adicionar os valores de AIC e BIC aos resultados
        df_results['AIC'] = modelo_arima0.aic()
        df_results['BIC'] = modelo_arima0.bic()
        df_results['store_id']=store_id
        df_results['Modelo'] = modelo
        store_data.reset_index(drop=True, inplace=True)
        df_results['store_type']=store_data['storetype_id'].loc[1]
        df_results['store_size']=store_data['store_size'].loc[1]
        df_results['city_code']= store_data['city_code'].loc[1]
        
        df_results_63=pd.concat([df_results_63,df_results],axis=0)
        
        
        #Plot Resíduos
        
        fig = plt.figure()
        modelo_arima0.plot_diagnostics(figsize=(10,8))
        #plt.savefig("".join([path5, 'Resíduos', store_id,'.png']))
        #plt.show()
        plt.close(fig)
        
        # Plotar os dados dos erros de estimação
        fig = plt.figure()
        plt.figure(figsize=(15, 12))
        plt.scatter(time_test_data, erros_estimacao)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Semanas',fontsize=(20))
        plt.ylabel('Erros de Estimação',fontsize=(20))
        plt.title('Gráfico de Erros de estimação do Modelo ARIMA',fontsize=(30))
        #plt.savefig("".join([path5, 'Erros', store_id,'.png']))
        #plt.show()
        plt.close(fig)
        
        # Plotar os dados originais, as previsões e os dados de teste
        fig = plt.figure()
        plt.figure(figsize=(15, 12))
        plt.plot(store_data['continuous_week_number'], store_data['revenue'], label='Dados Originais')
        plt.plot(time_test_data, forecast, label='Previsões')
        plt.plot(time_test_data, test_data['revenue'], label='Dados de Teste')
        plt.xlabel('Semanas',fontsize=(20))
        plt.ylabel('Revenue',fontsize=(20))
        plt.title('Previsões ARIMA',fontsize=(30))
        plt.legend()
        #plt.savefig("".join([path5, 'Previsão', store_id,'.png']))
        #plt.show()
        plt.close(fig)
    

        # Salvar o DataFrame no arquivo Excel
        #df_results_63.to_csv("".join([path5,'resultados_arima.csv']),sep=',',index=False)
        #df_forecast.to_csv("".join([path5,'resultados_forecast_st03_03.csv']),sep=',',index=False)
#Tratamento dos resultados sarimax

#filepath5 =  r"C:/Users/jpaul/OneDrive/Ambiente de Trabalho/Projeto2/Sarimax_revenues/resultados_arima.csv"

df_results_63 = pd.read_csv(filepath5)

valores_a_verificar = ['stock_inicial', 'Probin1_very_low', 'Probin1_low', 'Probin1_moderate', 'Probin1_high', 'Probin1_very_high', 'Probin2_very_low', 'Probin2_high', 'Probin2_very_high', 'promo_discount_16', 'promo_discount_20', 'promo_discount_35', 'promo_discount_40', 'promo_discount_50', 'Primavera', 'Verao', 'Outono', 'Feriados']

df_results_63_tratado = df_results_63.drop(df_results_63[df_results_63['Tipo de Coeficiente'].isin(valores_a_verificar) & (df_results_63['p_value'] > 0.1)].index)
df_results_63_tratado =df_results_63_tratado.dropna(subset=['p_value'])

df_results_63_tratado.to_csv("".join([path3,'resultados_Sarimax_tratado.csv']),sep=',',index=False)

#Contar número de vezes que aparece os tipos de coeficientes e criar duas colunas com o tipo de coeficiente e a contagem
contagem_tipo_de_coeficiente = df_results_63_tratado['Tipo de Coeficiente'].value_counts().reset_index()
contagem_tipo_de_coeficiente.columns = ['Tipo de Coeficiente', 'Contagem']


#Criar duas colunas com o min e maximo de todos os coeficientes
df_min_max = df_results_63_tratado.groupby('Tipo de Coeficiente')['Coeficiente'].apply(lambda x: (x.min(), x.max())).reset_index()
df_min_max[['Coeficiente_min', 'Coeficiente_max']] = pd.DataFrame(df_min_max['Coeficiente'].tolist(), index=df_min_max.index)
df_min_max = df_min_max.drop('Coeficiente', axis=1)

#Agrupar a contagem com o mínimo e o máximo
contagem_tipo_de_coeficiente = contagem_tipo_de_coeficiente.groupby('Tipo de Coeficiente').size().reset_index(name='Contagem')
contagem_tipo_de_coeficiente= pd.concat([contagem_tipo_de_coeficiente.set_index('Tipo de Coeficiente'), df_min_max.set_index('Tipo de Coeficiente')], axis=1, join='inner')

#Eliminar todas as linhas com tipos de coeficiente
valores_desejados = ['stock_inicial', 'Probin1_high', 'Probin1_very_high', 'Verao', 'Primavera', 'Outono', 'Feriados', 'Probin1_low', 'Probin1_very_low', 'Probin1_moderate', 'promo_discount_16']
contagem_tipo_de_coeficiente.reset_index(inplace=True)
contagem_tipo_de_coeficiente = contagem_tipo_de_coeficiente[contagem_tipo_de_coeficiente['Tipo de Coeficiente'].isin(valores_desejados)]

# Colunas arranjos casas decimais
contagem_tipo_de_coeficiente['Coeficiente_min'] = contagem_tipo_de_coeficiente['Coeficiente_min'].round(2)
contagem_tipo_de_coeficiente['Coeficiente_max'] = contagem_tipo_de_coeficiente['Coeficiente_max'].round(2)

# Plotar tabela
fig, ax = plt.subplots()
ax.axis('off')  # Desativar os eixos
ax.table(cellText=contagem_tipo_de_coeficiente.values, colLabels=contagem_tipo_de_coeficiente.columns, loc='center')

plt.show()

#contagem_tipo_de_coeficiente.to_csv("".join([path3,'tabela_range_coef_exog.csv']),sep=',',index=False)

#%%Sarimax lojas individual do cluster  sem as exógenas que não têm significância


#Para o sarimax das lojas individuais removeu-se algumas variaveis que não tinham significância
grouped_df= df_final.groupby(['store_id', 'continuous_week_number']).agg({
    'sales':'sum',
    'revenue': 'sum',
    'stock_inicial': 'max',
    'Probin1_very_low':'sum',
    'Probin1_low':'sum',
    'Probin1_moderate':'sum',
    'Probin1_high':'sum',
    'Probin1_very_high':'sum',
    'Primavera':'mean',
    'Verao':'mean',
    'Outono':'mean',
    'Feriados':'mean',
}).reset_index()
#grouped2_df.to_csv('Loja_Sales_Revenue2.csv', index=False)
df_forecast = pd.DataFrame()

valores_unicos = grouped_df['store_id'].unique().tolist()
df_results_63 = pd.DataFrame()
ST03_03 = ['S0141', 'S0120', 'S0077', 'S0143', 'S0068', 'S0039', 'S0016', 'S0080']
metrics_df = pd.DataFrame(columns=['store_id', 'MAE', 'MSE', 'RMSE', 'MAPE'])
for store_id in ST03_03:  
    #store_id='S0141'
    
    # Filtrar os dados para a loja específica
    store_data = grouped_df[grouped_df['store_id'] == store_id]
    store_data.index=range(0,len(store_data['revenue']))     
    tamanho = len(store_data['store_id'])
    
    if tamanho > 79:  
        inicio=store_data.index[0]
        
        # Definir o número da semana para o início e o fim do conjunto de treino e teste
        train_start_week = inicio # Definir o número da semana de início do treino
        train_end_week = inicio+len(store_data)-10 # Definir o número da semana de término do treino
        test_start_week = inicio+len(store_data)-9  # Definir o número da semana de início do teste
        test_end_week = inicio+len(store_data)-1   # Definir o número da semana de término do teste
        
        # Filtrar os dados para os períodos de treinamento e teste
        colunas_remover=['sales','continuous_week_number','store_id']
        dados_final=store_data.drop(colunas_remover, axis=1)
        train_data = dados_final.loc[train_start_week:train_end_week]
        time_train_data = store_data.loc[train_start_week:train_end_week,'continuous_week_number']
        test_data = dados_final.loc[test_start_week:test_end_week]
        time_test_data = store_data.loc[test_start_week:test_end_week,'continuous_week_number']
        
        # Ajustar o modelo ARIMA
        Exogenas_remover=['revenue']
        #,'promo_discount_16','promo_discount_20','promo_discount_35','promo_discount_40','promo_discount_50','Primavera','Verao','Outono'
        Exogenas=train_data.drop(Exogenas_remover, axis=1)
        #Exogenas=['stock_inicial']
        modelo_arima0 = pm.auto_arima(train_data['revenue'], X=Exogenas, supress_warnings=True,stepwise=True, seasonal=True, trace=True, m=52)
        modelo=modelo_arima0.fit(train_data['revenue'], X=Exogenas)
        # Fazer previsões no conjunto de teste
        print(train_data.dtypes)

        #forecast = modelo_arima.predict(n_periods=len(test_data))
        Exogenas_test=test_data.drop(Exogenas_remover, axis=1)
        forecast = modelo_arima0.predict(n_periods=len(test_data),X=Exogenas_test)
        coeficientes = modelo_arima0.params()
        
        #Indice do forecast direito
        forecast.index=test_data.index
        # Calcular os resíduos
        erros_estimacao = test_data['revenue'] - forecast
        residuos1=modelo_arima0.resid()
        modelo_arima_summary = modelo.summary()
        
        # Exibir as previsões para a loja
        print(f"Previsões para a loja {store_id}:\n{forecast}\n")
        #Armazenar previsões
        df_forecast[store_id] = forecast
        #coeficientes
        print(coeficientes)
        coeficientes_exogenas = modelo_arima0.params()[modelo_arima0.order[2]:]
        print(coeficientes_exogenas)
        #summary=evaluate_arima_model(test_data, forecast)
        print(modelo_arima_summary)
        
        evaluation_result = evaluate_arima_model(test_data['revenue'], forecast)

        mae = evaluation_result['MAE'].values[0]
        mse = evaluation_result['MSE'].values[0]
        rmse = evaluation_result['RMSE'].values[0]
        mape = evaluation_result['MAPE'].values[0]
        metrics_df = metrics_df.append({
        'store_id': store_id,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
        }, ignore_index=True)
        
        df_results=pd.DataFrame()
        # Criar um DataFrame com os resultados dos coeficientes e erros padrão
        df_results = pd.DataFrame({'Tipo de Coeficiente':coeficientes.index,'Coeficiente': coeficientes,'p_value': modelo_arima0.pvalues(), 'inter_conf_inf':modelo_arima0.conf_int()[0],'inter_conf_sup':modelo_arima0.conf_int()[1]})
    
        # Adicionar os valores de AIC e BIC aos resultados
        df_results['AIC'] = modelo_arima0.aic()
        df_results['BIC'] = modelo_arima0.bic()
        df_results['store_id']=store_id
        df_results['Modelo'] = modelo
        store_data.reset_index(drop=True, inplace=True)
        #df_results['store_type']=store_data['storetype_id'].loc[1]
        #df_results['store_size']=store_data['store_size'].loc[1]
        #df_results['city_code']= store_data['city_code'].loc[1]
        
        df_results_63=pd.concat([df_results_63,df_results],axis=0)
        
        
        #Plot Resíduos
        
        fig = plt.figure()
        modelo_arima0.plot_diagnostics(figsize=(10,8))
        #plt.savefig("".join([path5, 'Resíduos', store_id,'.png']))
        #plt.show()
        plt.close(fig)
        
        # Plotar os dados dos erros de estimação
        fig = plt.figure()
        plt.figure(figsize=(15, 12))
        plt.scatter(time_test_data, erros_estimacao)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Semanas',fontsize=(20))
        plt.ylabel('Erros de Estimação',fontsize=(20))
        plt.title('Gráfico de Erros de estimação do Modelo ARIMA',fontsize=(30))
        #plt.savefig("".join([path5, 'Erros', store_id,'.png']))
        #plt.show()
        plt.close(fig)
        
        # Plotar os dados originais, as previsões e os dados de teste
        fig = plt.figure()
        plt.figure(figsize=(15, 12))
        plt.plot(store_data['continuous_week_number'], store_data['revenue'], label='Dados Originais')
        plt.plot(time_test_data, forecast, label='Previsões')
        plt.plot(time_test_data, test_data['revenue'], label='Dados de Teste')
        plt.xlabel('Semanas',fontsize=(20))
        plt.ylabel('Revenue',fontsize=(20))
        plt.title('Previsões ARIMA',fontsize=(30))
        plt.legend()
        #plt.savefig("".join([path5, 'Previsão', store_id,'.png']))
        #plt.show()
        plt.close(fig)
    

        # Salvar o DataFrame no arquivo Excel
        #df_results_63.to_csv("".join([path5,'resultados_arima.csv']),sep=',',index=False)
        #df_forecast.to_csv("".join([path5,'resultados_forecast_st03_03.csv']),sep=',',index=False)
metrics_df.to_csv("".join([path5,'resultados_metrics_st03_03.csv']),sep=',',index=False)
#Tratamento dos resultados sarimax

#%%Sarimax para clusters
grouped_df= df_final.groupby(['store_id', 'continuous_week_number']).agg({
    'revenue': 'sum',
    'stock_inicial': 'max',
    'Probin1_very_low':'sum',
    'Probin1_low':'sum',
    'Probin1_moderate':'sum',
    'Probin1_high':'sum',
    'Probin1_very_high':'sum',
    'Primavera':'mean',
    'Verao':'mean',
    'Outono':'mean',
    'Feriados':'mean'
}).reset_index()

ST03_03 = ['S0141', 'S0120', 'S0077', 'S0143', 'S0068', 'S0039', 'S0016', 'S0080']
df_cluster = df_final[df_final['store_id'].isin(ST03_03)].copy()


store_data= df_cluster.groupby(['continuous_week_number']).agg({
    'revenue': 'sum',
    'stock_inicial': 'sum',
    'Probin1_very_low':'sum',
    'Probin1_low':'sum',
    'Probin1_moderate':'sum',
    'Probin1_high':'sum',
    'Probin1_very_high':'sum',
    'Primavera':'mean',
    'Verao':'mean',
    'Outono':'mean',
    'Feriados':'mean',
}).reset_index()

#%%Transformar dados do cluster

#Remover colunas que não podem ser transformadas
Store_data_sem_colunas = store_data.drop(['continuous_week_number','Primavera', 'Verao', 'Outono', 'Feriados'], axis=1)
# Crie uma instância do StandardScaler
scaler = StandardScaler()
# Ajustar o scaler aos dados de treinamento
scaler.fit(Store_data_sem_colunas)
# Aplicar escalonamento aos dados do cluster
store_data_scaled = scaler.transform(Store_data_sem_colunas)
#Voltar a nomear as colunas
store_data_scaled = pd.DataFrame(store_data_scaled, columns=Store_data_sem_colunas.columns)

#Juntar de voltas as colunas removidas
# Selecionar as colunas desejadas do DataFrame original
colunas_desejadas = ['continuous_week_number','Primavera', 'Verao', 'Outono', 'Feriados']
store_data_selected = store_data[colunas_desejadas]

# Juntar as colunas selecionadas ao DataFrame escalado
store_data = pd.concat([store_data_scaled, store_data_selected], axis=1)


df_results_63 = pd.DataFrame()
# DataFrame para armazenar as previsões
df_forecast = pd.DataFrame(index=test_data.index)
#Dataframe para MAE, MSE, RMSE e MAPE
metrics_df = pd.DataFrame(columns=['store_id', 'MAE', 'MSE', 'RMSE', 'MAPE'])

for store_id in ST03_03:
      
    #store_id='S0141'
    store_data_lojas = grouped_df[grouped_df['store_id'] == store_id]
    store_data_lojas.index=range(0,len(store_data_lojas['revenue']))     
    
    #%%Transformar dados de cada loja
    
    
    #Remover colunas que não podem ser transformadas
    Store_data_lojas_sem_colunas = store_data_lojas.drop(['store_id','continuous_week_number','Primavera', 'Verao', 'Outono', 'Feriados'], axis=1)
    # Crie uma instância do StandardScaler
    scaler = StandardScaler()
    # Ajustar o scaler aos dados de treinamento
    scaler.fit(Store_data_lojas_sem_colunas)
    # Aplicar escalonamento aos dados do cluster
    store_data_lojas_scaled = scaler.transform(Store_data_lojas_sem_colunas)
    #Voltar a nomear as colunas
    store_data_lojas_scaled = pd.DataFrame(store_data_lojas_scaled, columns=Store_data_lojas_sem_colunas.columns)

    #Juntar de voltas as colunas removidas
    # Selecionar as colunas desejadas do DataFrame original
    colunas_desejadas = ['store_id','continuous_week_number','Primavera', 'Verao', 'Outono', 'Feriados']
    store_data_lojas_selected = store_data_lojas[colunas_desejadas]

    # Juntar as colunas selecionadas ao DataFrame escalado
    store_data_lojas_final = pd.concat([store_data_lojas_scaled, store_data_lojas_selected], axis=1)


    #%%Modelação 
    inicio=store_data_lojas.index[0]
    
    # Definir o número da semana para o início e o fim do conjunto de treino e teste
    #train_start_week = inicio # Definir o número da semana de início do treino
    #train_end_week = inicio+len(store_data)-10 # Definir o número da semana de término do treino
    test_start_week = inicio+len(store_data_lojas)-9  # Definir o número da semana de início do teste
    test_end_week = inicio+len(store_data_lojas)-1   # Definir o número da semana de término do teste
    
    # Filtrar os dados para os períodos de treinamento e teste
    dados_reais=store_data_lojas.loc[test_start_week:test_end_week]
    colunas_remover=['continuous_week_number']
    colunas_remover_lojas=['continuous_week_number', 'store_id']
    dados_final=store_data.drop(colunas_remover, axis=1)
    dados_final_lojas=store_data_lojas_final.drop(colunas_remover_lojas, axis=1)
    train_data = dados_final
    time_train_data = store_data['continuous_week_number']
    test_data = dados_final_lojas.loc[test_start_week:test_end_week]
    time_test_data = store_data_lojas_final.loc[test_start_week:test_end_week,'continuous_week_number']
    
    # Ajustar o modelo ARIMA
    Exogenas_remover=['revenue']
    Exogenas=train_data.drop(Exogenas_remover, axis=1)
    modelo_arima0 = pm.auto_arima(train_data['revenue'], X=Exogenas, supress_warnings=True,stepwise=True, seasonal=True, trace=True, m=52)
    modelo=modelo_arima0.fit(train_data['revenue'], X=Exogenas)
    # Fazer previsões no conjunto de teste
    print(train_data.dtypes)
    
    #forecast = modelo_arima.predict(n_periods=len(test_data))
    Exogenas_test=test_data.drop(Exogenas_remover, axis=1)
    forecast = modelo_arima0.predict(n_periods=len(test_data),X=Exogenas_test)
    
    #%% Transformar forecast
    #Passar forecast para data frame
    forecast = pd.DataFrame(forecast)
    forecast.set_index(test_data.index, inplace=True)
    #juntar tabelas
    Revenue_previsão=test_data.drop(['revenue','Primavera', 'Verao', 'Outono', 'Feriados'], axis=1)
    # Juntar as colunas selecionadas ao DataFrame escalado
    forecast = pd.concat([forecast, Revenue_previsão], axis=1)
    #Tranformação dos valores
    forecast= scaler.inverse_transform(forecast)
    #Voltar a nomear as colunas
    forecast = pd.DataFrame(forecast, columns=store_data_scaled.columns)

    
    coeficientes = modelo_arima0.params()
    
    #Indice do forecast direito
    forecast.index=test_data.index
    # Calcular os resíduos
    erros_estimacao = dados_reais['revenue'] - forecast['revenue']
    residuos1=modelo_arima0.resid()
    modelo_arima_summary = modelo.summary()
    
    # Exibir as previsões para a loja
    print(f"Previsões para a loja {store_id}:\n{forecast}\n")
    #Armazenar previsões
    df_forecast[store_id] = forecast['revenue']
    #coeficientes
    print(coeficientes)
    coeficientes_exogenas = modelo_arima0.params()[modelo_arima0.order[2]:]
    print(coeficientes_exogenas)
    #summary=evaluate_arima_model(test_data, forecast)
    print(modelo_arima_summary)
    
    evaluation_result = evaluate_arima_model(dados_reais['revenue'], forecast['revenue'])

    mae = evaluation_result['MAE'].values[0]
    mse = evaluation_result['MSE'].values[0]
    rmse = evaluation_result['RMSE'].values[0]
    mape = evaluation_result['MAPE'].values[0]
    metrics_df = metrics_df.append({
    'store_id': store_id,
    'MAE': mae,
    'MSE': mse,
    'RMSE': rmse,
    'MAPE': mape
    }, ignore_index=True)
    
    df_results=pd.DataFrame()
    # Criar um DataFrame com os resultados dos coeficientes e erros padrão
    df_results = pd.DataFrame({'Tipo de Coeficiente':coeficientes.index,'Coeficiente': coeficientes,'p_value': modelo_arima0.pvalues(), 'inter_conf_inf':modelo_arima0.conf_int()[0],'inter_conf_sup':modelo_arima0.conf_int()[1]})
    
    # Adicionar os valores de AIC e BIC aos resultados
    df_results['AIC'] = modelo_arima0.aic()
    df_results['BIC'] = modelo_arima0.bic()
    df_results['store_id']=store_id
    df_results['Modelo'] = modelo
    store_data.reset_index(drop=True, inplace=True)
    #df_results['store_type']=store_data['storetype_id'].loc[1]
    #df_results['store_size']=store_data['store_size'].loc[1]
    #df_results['city_code']= store_data['city_code'].loc[1]
    
    df_results_63=pd.concat([df_results_63,df_results],axis=0)
    
    
    #Plot Resíduos
    
    fig = plt.figure()
    modelo_arima0.plot_diagnostics(figsize=(10,8))
    plt.savefig("".join([path4, 'Resíduos', store_id,'.png']))
    #plt.show()
    plt.close(fig)
    
    # Plotar os dados dos erros de estimação
    fig = plt.figure()
    plt.figure(figsize=(15, 12))
    plt.scatter(time_test_data, erros_estimacao)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Semanas',fontsize=(20))
    plt.ylabel('Erros de Estimação',fontsize=(20))
    plt.title('Gráfico de Erros de estimação do Modelo ARIMA',fontsize=(30))
    plt.savefig("".join([path4, 'Erros', store_id,'.png']))
    #plt.show()
    plt.close(fig)
    
    #%% Plotar os dados originais, as previsões e os dados de teste
    
    
    fig = plt.figure()
    plt.figure(figsize=(15, 12))
    plt.plot(dados_reais['continuous_week_number'], dados_reais['revenue'], label='Dados Originais')
    plt.plot(time_test_data, forecast['revenue'], label='Previsões')
    plt.plot(time_test_data, dados_reais['revenue'], label='Dados de Teste')
    plt.xlabel('Semanas',fontsize=(20))
    plt.ylabel('Revenue',fontsize=(20))
    plt.title('Previsões ARIMA',fontsize=(30))
    plt.legend()
    plt.savefig("".join([path4, 'Previsão', store_id,'.png']))
    #plt.show()
    plt.close(fig)
    
    
    # Salvar o DataFrame no arquivo Excel
    df_results_63.to_csv("".join([path4,'resultados_arima.csv']),sep=',',index=False)
    df_forecast.to_csv("".join([path4,'resultados_forecast_st03_03.csv']),sep=',',index=False)
    metrics_df.to_csv("".join([path4,'resultados_metrics_st03_03.csv']),sep=',',index=False)




