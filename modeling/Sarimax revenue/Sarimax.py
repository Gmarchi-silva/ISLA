# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 21:02:01 2023

@author: jpaul
"""

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
df_forecast = pd.DataFrame()

for store_id in valores_unicos:  
    store_id='S0141'
    
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
#df_results_63.to_csv("".join([path1,'resultados_arima.csv']),sep=',',index=False)
#df_forecast.to_csv("".join([path1,'resultados_forecast_st03_03.csv']),sep=',',index=False)


#%%Tratamento dos resultados sarimax

#filepath2 =  r"C:/Users/jpaul/OneDrive/Ambiente de Trabalho/Projeto2/Sarimax_revenues/resultados_arima.csv"

#df_results_63 = pd.read_csv(filepath2)

valores_a_verificar = ['stock_inicial', 'Probin1_very_low', 'Probin1_low', 'Probin1_moderate', 'Probin1_high', 'Probin1_very_high', 'Probin2_very_low', 'Probin2_high', 'Probin2_very_high', 'promo_discount_16', 'promo_discount_20', 'promo_discount_35', 'promo_discount_40', 'promo_discount_50', 'Primavera', 'Verao', 'Outono', 'Feriados']

df_results_63_tratado = df_results_63.drop(df_results_63[df_results_63['Tipo de Coeficiente'].isin(valores_a_verificar) & (df_results_63['p_value'] > 0.1)].index)
df_results_63_tratado =df_results_63_tratado.dropna(subset=['p_value'])

#df_results_63_tratado.to_csv("".join([path1,'resultados_Sarimax_tratado.csv']),sep=',',index=False)

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

#contagem_tipo_de_coeficiente.to_csv("".join([path1,'tabela_range_coef_exog.csv']),sep=',',index=False)
