# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 21:12:40 2023

@author: jpaul
"""

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



def evaluate_arima_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    summary = pd.DataFrame({'MAE': [mae], 'MSE': [mse], 'RMSE': [rmse], 'MAPE': [mape]})
    return summary

path1="C:/Users/jpaul/OneDrive/Ambiente de Trabalho/Projeto2/"


#%%2 Importar ficheiro CSV

filepath =  r'C:/Users/jpaul/OneDrive/Ambiente de Trabalho/Projeto2/Data_Preparation/df_final.csv'

df_final = pd.read_csv(filepath)
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
df_forecast = pd.DataFrame()
#Dataframe para MAE, MSE, RMSE e MAPE
metrics_df = pd.DataFrame(columns=['store_id', 'MAE', 'MSE', 'RMSE', 'MAPE'])

for store_id in ST03_03:
      
    
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
    train_start_week = inicio # Definir o número da semana de início do treino
    train_end_week = inicio+len(store_data)-10 # Definir o número da semana de término do treino
    test_start_week = inicio+len(store_data_lojas)-9  # Definir o número da semana de início do teste
    test_end_week = inicio+len(store_data_lojas)-1   # Definir o número da semana de término do teste
    
    # Filtrar os dados para os períodos de treinamento e teste
    dados_reais=store_data_lojas.loc[test_start_week:test_end_week]
    colunas_remover=['continuous_week_number']
    colunas_remover_lojas=['continuous_week_number', 'store_id']
    dados_final=store_data.drop(colunas_remover, axis=1)
    dados_final_lojas=store_data_lojas_final.drop(colunas_remover_lojas, axis=1)
    train_data = dados_final.loc[train_start_week:train_end_week]
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
    #plt.savefig("".join([path4, 'Resíduos', store_id,'.png']))
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
    #plt.savefig("".join([path4, 'Erros', store_id,'.png']))
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
    #plt.savefig("".join([path4, 'Previsão', store_id,'.png']))
    #plt.show()
    plt.close(fig)
    
    
# Salvar o DataFrame no arquivo Excel
#df_results_63.to_csv("".join([path4,'resultados_arima.csv']),sep=',',index=False)
#df_forecast.to_csv("".join([path4,'resultados_forecast_st03_03.csv']),sep=',',index=False)
#metrics_df.to_csv("".join([path4,'resultados_metrics_st03_03.csv']),sep=',',index=False)

