# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 22:09:34 2023

@author: igort
"""

#%% Importar as bibliotecas necessárias

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

path1 = "C:/Users/igort/Documents/Igor/ISLA GAIA/14.Projeto II/Drive/"

path2 = "C:/Users/igort/Documents/Igor/ISLA GAIA/14.Projeto II/Resultados/"

#%% Média
grouped2_df = pd.read_csv(r'C:/Users/igort/Documents/Igor/ISLA GAIA/14.Projeto II/Drive/grouped2_df.csv')

contagem = grouped2_df['store_id'].value_counts()

valores_unicos = contagem.index.tolist()


for store_id in valores_unicos:
  
    # Filtrar os dados para a loja específica
    store_data = grouped2_df[grouped2_df['store_id'] == store_id]

    inicio=store_data.index[1]
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
         
    # Calcular a média do treino
    mean_train_data = train_data.mean()
    
    # Criar as previsões com base na média do treino
    prev_media_revenue = pd.Series(mean_train_data, index=test_data.index)
       
    
    # Avaliar o desempenho das previsões (opcional)
    r2_rev = r2_score(test_data, prev_media_revenue)
    mae_rev = mean_absolute_error(test_data, prev_media_revenue)
    rmse_rev = np.sqrt(mean_squared_error(test_data, prev_media_revenue))
   
    # Criar um DataFrame com os resultados
    df_results_media = pd.DataFrame({'Tipo de Coeficiente': ['na'],
                                    'store_id': store_id,
                                    'Modelo': 'media',
                                    'store_type': store_data['storetype_id'].iloc[1],
                                    'store_size': store_data['store_size'].iloc[1],
                                    'city_code': store_data['city_code'].iloc[1]})
   
    
    # Calcular a soma dos resíduos quadrados (SSE)
    sse = ((test_data - prev_media_revenue) ** 2).sum() 
    
    # Calcular AIC e BIC
    n = len(test_data)
    
    k = 1  # Número de parâmetros no modelo - média
    
    aic = n * np.log(sse / n) + 2 * k
    
    bic = n * np.log(sse / n) + k * np.log(n)
    
    print("AIC:", aic)
    print("BIC:", bic)
    
   # Adicionar os resultados ao DataFrame
    df_results_media['SSE'] = sse
    df_results_media['AIC'] = aic
    df_results_media['BIC'] = bic
    
    # Adicionar os resultados ao DataFrame principal
    df_results_media_63 = pd.concat([df_results_media_63, df_results_media], ignore_index=True)

    
    
# Salvar o DataFrame num arquivo
df_results_media_63.to_csv("".join([path2,'resultados_media_revenues.csv']),sep=',',index=False)
