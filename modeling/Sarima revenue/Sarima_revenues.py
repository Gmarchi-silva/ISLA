# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 18:57:50 2023

@author: jpaul
"""

import numpy as np
import pandas as pd
import pmdarima as pm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import math
import re
import openpyxl




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

#%% SARIMA e ARIMA (com auto_arima)

#%Agrupar os dados
grouped_df= df_final.groupby(['store_id', 'continuous_week_number','storetype_id','city_code']).agg({
    'sales': 'sum',
    'revenue': 'sum',
    'store_size': 'mean'
    
}).reset_index()


df_results_63 = pd.DataFrame()
valores_unicos = grouped_df['store_id'].unique().tolist()
df_forecast=pd.DataFrame()

for store_id in valores_unicos:

    
    #store_id='S0002'
    
    
    # Filtrar os dados para a loja específica
    store_data = grouped_df[grouped_df['store_id'] == store_id]
     
    tamanho = len(store_data['store_id'])
    #Colocamos maior~que 79 semanas uma vez que a loja que está logo abaixo das 79 semanas dá erro a correr um arima com sazonalidade
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
