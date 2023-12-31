# Projeto II

## 1.	Enquadramento / Business Understanding

Foi disponibilizada informação de uma empresa cuja atividade é de comércio de roupa em lojas físicas presentes em diversas cidades da Turquia.
A empresa precisa de atualizar o armazenamento de roupa em cada loja para Outubro de 2019 por isso o objetivo principal passa por prever com precisão as vendas semanais de cada loja considerando  sazonalidade, tendência e variáveis explicativas complementando com uma análise dos dados que serão utilizados no modelo.
Neste sentido, e seguindo a metodologia CRISP-DM, pretendemos criar modelo (s) de previsão para as lojas e prever as vendas para Outubro de 2019 para cada loja, avaliando se a previsão é próxima do número de vendas real para que a gestão do armazenamento possa ser o mais eficiente possível e potenciar o aumento das vendas/receitas.

## 2.	Data Understanding

Os dados disponibilizados são em formato “csv” e correspondem a três tabelas ('sales', ['product'](https://github.com/Gmarchi-silva/ISLA/blob/main/Raw_Data/product.csv) e ['cities'](https://github.com/Gmarchi-silva/ISLA/blob/main/Raw_Data/cities.csv)).
Para a análise de dados utilizamos as ferramentas Excel, Python (Spyder e Google Colab) e Rstudio para extrair o melhor entendimento possível da informação presente nos dados.
Primeiro verificamos as variáveis presentes em cada uma das tabelas e o possível relacionamento existente entre elas. Neste caso, as tabelas “cities” e “product” estão relacionadas diretamente com a tabela “sales” pelas variáveis “store_id” e “product_id” respectivamente, podendo estas colunas serem consideradas chaves primárias visto que apenas contêm dados únicos (sendo chaves estrangeiras na tabela “sales”).

Para apoiar nesta análise recorremos à biblioteca “pandas_profiling” do Python (relatórios de cada tabela no repositório) e percebemos o seguinte:
  
  ### Tabela cities
  - A [Tabela Cities](https://htmlpreview.github.io/?https://github.com/Gmarchi-silva/ISLA/blob/main/Pandas%20Profiling/1_p_profiling_cities.html)  contém 6 variáveis (1 numérica e 6 categóricas) e 63 observações, sem valores em falta:
  - Existem 63 lojas classificadas em 4 tipos e com 32 tamanhos entre elas sendo que nos tipos das lojas conseguimos perceber que são pequenas, médias e grandes, havendo uma especial pois é a única presente no tipo ST02.
  - As lojas são na Turquia e dispersas por 19 cidades diferentes, no entanto, 32 delas estão na cidade de Istanbul.

  ### Tabela Product
  - A [Tabela Product](https://htmlpreview.github.io/?https://github.com/Gmarchi-silva/ISLA/blob/main/Pandas%20Profiling/1_p_profiling_product.html) contém 10 variáveis (3 numéricas e 7 categóricas) e 699 observações, com 100 valores em falta (1,4%)
- Existem registados 699 produtos diferentes segmentados em 10 clusters pela coluna “cluster_id” (havendo 50 produtos sem segmento e sendo o “cluster_0” o mais representativo com 450 produtos – 64,4%) que não conseguimos perceber com estes dados os critérios.
- As colunas “product_length”, “product_depth” e “product_width” caracterizam as dimensões do produto e possuem valores em falta (nem todos comuns às 3 colunas) e um registo zero.
- As 5 colunas “hierarchy…” classificam os produtos em vários níveis e nenhuma possui valores em falta.

  ### Tabela Sales
- A [Tabela Sales](https://htmlpreview.github.io/?https://github.com/Gmarchi-silva/ISLA/blob/main/Pandas%20Profiling/1_p_profiling_sales.html) contém 14 variáveis (6 numéricas e 8 categóricas) e 8.886.058 observações, com 35.271.795 valores em falta (28,4%), mas sem duplicados.
	- Possui uma coluna com números sequenciais e que não se repetem.
	- “store_id” com registos de todas as 63 lojas.
	- “product_id” com registos de apenas 615 produtos dos 699 registados na tabela “product”.
	- “date” com datas compreendidas entre o dia 02-01-2017 e 31-10-2019.
- “sales” com as quantidades vendidas (mínimo de 0 e máximo de 43301), 3,4% de valores em falta e 79,3% de zeros
- “revenue” com a receita da venda, 3,4% de valores em falta (igual a “sales”) e 79,3% de zeros (1072 zeros a mais que “sales”)
- “stock”com 3,4% de valores em falta(igual a “sales” e “revenue”) e 0,7% de zeros.
- “price” com 606 valores distintos (ou seja, existem produtos com o mesmo preço) e 91381 valores em falta (1972 não apresentam valores em “sales”, “revenue” e “stock” | 69035 com “sales”=0 e “revenue”=0 | 69057 com “revenue”=0 | 22 com “revenue”=0 mas “sales”>0 de produtos, lojas e datas diferentes)
- Características das variáveis “promo”:
•	“type_1” com 17 tipos sendo a PR14 a que se destaca com 86,1% das vendas
•	“bin_1” com 5 características mas havendo 86,1% das vendas sem nada registado
•	“type_2” com 4 tipos sendo 99,9% dos valores do tipo PR03
•	“bin_2” com 3 características mas 99,9% das vendas sem registo
•	“discount_2” com 6 valores diferentes de desconto atribuídos às vendas com registo de “promo_type_2”/ “promo_bin_2” (ou seja, em 99,9% dos registos não tem valor)
•	“discount_type_2” com uma classificação em 4 tipos dos descontos/ “promo_2” anteriormente indicados (mantém os 99,9% de registos sem valor)

  Correlações:
- As ‘sales’ têm uma forte correlação positiva com a ‘revenue’, o que já seria de esperar
- A correlação de ‘sales’ com ‘stock’ é positiva mas pouco significativa
- O mesmo acontece com ‘sales’ e ‘price’ mas neste caso com correlação negativa
- A correlação entre ‘stock’ e ‘price’ é negativa mas não muito significativa
  
![Logo do GitHub](https://github.com/Gmarchi-silva/ISLA/blob/main/Pandas%20Profiling/Correla%C3%A7%C3%A3o.png)


## 3.	Data Preparation

 Para o tratamento e agregação dos dados todos que escolhemos foi utilizado o python [Data preparation](https://github.com/Gmarchi-silva/ISLA/blob/main/Data%20Preparation/Data_preparation.py) :
 
- Correção dos nomes  das cidades ("?" - substituir por "i") 
- Preencher os dados em falta nas colunas “promo_bin_1”, “promo_bin_2” e “promo_discount_type_2” com “NA” e na coluna “promo_discount_2” com “zero”.
- Agregação dos dados das 3 tabelas e eliminação das variáveis que não vamos utilizar: “city_id_old”, “Unnamed: 0”, “country_id”, “hierarchy1_id”, “hierarchy2_id”, “hierarchy3_id”, “hierarchy4_id”, “hierarchy5_id”, “product_length”, “product_depth”, “product_width”, “cluster_id”
- Verificação de nulos e eliminação das linhas que tinham nulos de “sales”, “revenue”, “stock”.
- Criação de uma coluna com o número contínuo de semanas com base na data.
- Eliminação da semana nº144 (última) por apenas ter dados de 1 dia

### Lojas anormais

Como temos lojas com valores de semana muito abaixo de 143 decidimos fazer uma análise particular.

![Logo do GitHub](https://github.com/Gmarchi-silva/ISLA/blob/main/Data%20Preparation/Lojas%20anormais%20totais.png)

Verificamos que algumas lojas destas têm o primeiro registo de vendas na semana 53 e depois têm um período sem vendas (sendo esse período diferente de loja para loja) o que nos leva a crer que o primeiro registo se tratou de um teste e que as semanas seguintes sem registos se deveram à preparação da loja para abertura definitiva e por isso eliminamos a semana 53 dessas lojas e utilizamos os dados apenas das semanas seguintes que tinham registos.
Existem também 2 lojas (S0007 e S0059) que começaram a vender em semanas diferentes pelo que utilizamos os dados apenas das semanas seguintes que tinham registos.

![Logo do GitHub](https://github.com/Gmarchi-silva/ISLA/blob/main/Data%20Preparation/Lojas%20anormais.png)


Temos ainda o caso especial da loja S0136 que percebemos que não está aberta todo o ano e, pela análise efetuada a loja terá fechado em setembro como habitualmente fez nos períodos homólogos de 2017 e 2018, pelo que a previsão para outubro de 2019 é que esteja fechada e por isso não haja vendas.

![Logo do GitHub](https://github.com/Gmarchi-silva/ISLA/blob/main/Data%20Preparation/Lojas%20S0136.png)



### Criação das variaveis exógenas

Foram criadas várias variáveis exógenas para treino no modelo em sarimax, as variáveis foram:
- Stock_inicial: calculada pela soma do 'sales' com o 'stock' diário. Para cada semana utilizou-se o dia com o máximo diário superior.
- Promo bin 1 (very low, low, moderate, high, very high): neste caso somou-se o número de produtos que tiveram estas promoções por dia e por fim semanalmente.
- Promo bin 2 (very low, high, very high): neste caso somou-se o número de produtos que tiveram estas promoções por dia e por fim semanalmente.
- promo_discount (16, 20, 35, 40, 50): neste caso somou-se o número de produtos que tiveram estas promoções por dia e por fim semanalmente.
- Outono, Verão e Primavera: para estas variáveis foram criadas colunas 'dummy' para cada, em que foi colocado o número '1' caso a semana estivesse naquela estação do ano e '0' se não estivesse.
- Feriados: neste caso foi criada uma só coluna em que '1' representava uma semana com feriado e '0' se não tivesse. Nota: Se o feriado ficasse no início da semana colocava '1' também na semana anterior e se ficasse no final da semana colocava '1' na semana seguinte.

Depois deste tratamento dos dados agregamos tudo num dataset e “csv” final [df_final](https://github.com/Gmarchi-silva/ISLA/blob/main/Data%20Preparation/df_final.csv).

## 4.	Modeling

### Média e Média móvel
Para a criação de modelos, iniciamos com a [média](https://github.com/Gmarchi-silva/ISLA/blob/main/Modelos%20Gerais%20Base/3_Modelo%20m%C3%A9dia_v2.py) e [média móvel](https://github.com/Gmarchi-silva/ISLA/blob/main/Modelos%20Gerais%20Base/3_Modelo%20m%C3%A9dia%20m%C3%B3vel%201trim%20(13%20semanas).py) (1 trimestre –> 13 semanas) que testamos para algumas lojas e analisamos os resultados com base no R2, MAE e RMSE, no entanto, tendo em conta os resultados, e considerando que iremos avançar para um modelo ARIMA, que já nos dará indicação de modelos Auto-Regressivos e de Média Móvel, assim como o tempo disponível, resolvemos começar com a criação de modelo ARIMA com a configuração automática dos parâmetros (auto-arima) e avaliar os resultados.

### Arima Manual
De seguida começamos a efetuar o modelo [ARIMA Manual](https://github.com/Gmarchi-silva/ISLA/blob/main/Modelos%20Gerais%20Base/3_Modelo%20arima%20manual.R) com cálculo e definição manual de parâmetros, no entanto, esta opção implica efetuar manualmente por loja e mais uma vez tendo o tempo limitado optamos por seguir com ARIMA utilizando o auto-arima para definição dos parâmetros automaticamente. 

### Arima ([sales](https://github.com/Gmarchi-silva/ISLA/blob/main/modeling/Arima%20sales/Arima_sales.py) e [revenue](https://github.com/Gmarchi-silva/ISLA/blob/main/modeling/Arima%20revenues/Arima_revenues.py))
Corremos os modelos utilizando as variáveis [“sales”](https://github.com/Gmarchi-silva/ISLA/tree/main/modeling/Arima%20sales) e [“revenue”](https://github.com/Gmarchi-silva/ISLA/tree/main/modeling/Arima%20revenues) de forma a comparar qual delas poderia ser a melhor a utilizar para o fim proposto. 

-Os resultados do arima sales relativo a coeficientes, aic, bic,... encontram-se [aqui](https://github.com/Gmarchi-silva/ISLA/blob/main/modeling/Arima%20sales/resultados_arima_sales.csv).

-Os resultados do arima revenue relativo a coeficientes, aic, bic,... encontram-se [aqui](https://github.com/Gmarchi-silva/ISLA/blob/main/modeling/Arima%20revenues/resultados_arima_revenue.csv).

Nos restantes modelos optamos por nos centrarmos na “revenue”.


### Sarima ([revenue](https://github.com/Gmarchi-silva/ISLA/blob/main/modeling/Sarima%20revenue/Sarima_revenues.py))
Depois fizemos um sarima da ["revenue"](https://github.com/Gmarchi-silva/ISLA/tree/main/modeling/Sarima%20revenue) para todas as lojas para avaliar se a sazonalidade é relevante para todas as lojas individualmente. Neste caso de referir que fizemos um ciclo em que as lojas com menos de 79 semanas em vez de um sarima faziam um arima. Foi necessário fazer isto pois com a sazonalidade de 52 semanas o código dava erro para valores baixos. As lojas que ficaram de fora do sarima foram S0076, S0071, S0092, S0109 e S0007.
Os resultados do Sarima revenue relativo a coeficientes, aic, bic,... encontram-se [aqui](https://github.com/Gmarchi-silva/ISLA/blob/main/modeling/Sarima%20revenue/resultados_Sarima_revenues.csv).
 
### Sarimax ([revenue](https://github.com/Gmarchi-silva/ISLA/blob/main/modeling/Sarimax%20revenue/Sarimax.py))
 
Corremos o modelo Sarimax da ["revenue"](https://github.com/Gmarchi-silva/ISLA/tree/main/modeling/Sarimax%20revenue) com as seguintes variáveis exógenas:

- Stock_inicial 
- Promo bin 1 (very low, low, moderate, high, very high)
- Promo bin 2 (very low, high, very high)
- promo_discount (16, 20, 35, 40, 50)
- Outono, Verão e Primavera
- Feriados

Os resultados do Sarimax revenue relativo a coeficientes, aic, bic,... encontram-se [aqui](https://github.com/Gmarchi-silva/ISLA/blob/main/modeling/Sarimax%20revenue/resultados_Sarimax_com_clusters). Com estes dados decidimos criar clusters com base na store_type e nos modelos gerados pelo modelo SARIMAX para cada loja agrupando por store_type e de seguida por modelo gerado e identificamos 36 clusters diferentes havendo a possibilidade de agrupar lojas do mesmo tipo e com o mesmo modelo. 
A análise gráfica dos erros de cada loja parece revelar uma tendência para a sub-estimação da 'revenue', sendo relevante na análise final das previsões e tomada de decisão quanto ao armazenamento a ser efectuado, ressalvando que em praticamente todas as lojas existem semanas sub e sobre estimadas e ainda que na semana 138 existe um pico que foge bastante à tendência em praticamente todas as lojas.

Os mesmos dados permitiram-nos avaliar as variáveis exógenas, sendo que consideramos significativas as variáveis que tivessem um p-value < 0,10. Olhando para a tabela abaixo e analisando os dados obtidos percebemos que:

- O stock inicial é a variável que impacta em mais lojas, de todos os tipos, tamanhos e cidades, mas com coeficientes reduzidos.
- A Primavera e o Verão são as que causam um efeito de maior amplitude na 'revenue', e essencialmente na cidade de Istanbul, sendo o efeito negativo o de maior amplitude.
- Não têm impacto relevante na única loja do tipo ST02.
- O impacto, significativo, restringe-se a 7 cidades mas que são geograficamente dispersas.
- A maioria das lojas onde tem efeito são de tamanho pequeno e médio/baixo, no entanto, o maior impacto negativo é nas lojas de tamanho maior.
- As variáveis Probin1_low, Probin1_very_low, Outono, Probin1_moderate são as que causam apenas impacto positivo nas lojas também de vários tipos, tamanhos e cidades, em que foram consideradas significativas.

![image](https://github.com/Gmarchi-silva/ISLA/assets/125706061/3613aee0-06ca-4c49-8b6d-798791805f1e)

### Sarimax Cluster ([revenue](https://github.com/Gmarchi-silva/ISLA/blob/main/modeling/Sarimax%20cluster/Sarimax_cluster.py)) e lojas individuais ([revenue](https://github.com/Gmarchi-silva/ISLA/blob/main/modeling/Sarimax_lojas_STO3_03_indv/Sarimax_ST03_03.py))

Depois de analisarmos os dados do Sarimax percebemos que o cluster que queríamos utilizar era o ST03_03 que é composto por 8 lojas ()
A análise do Sarimax individual permitiu-nos também perceber que algumas variáveis não são significativas e por isso retiramos do modelo. Testámos o sarimax num [cluster](https://github.com/Gmarchi-silva/ISLA/tree/main/modeling/Sarimax%20cluster) específico e comparamos com os resultados do [modelo das lojas individuais](https://github.com/Gmarchi-silva/ISLA/tree/main/modeling/Sarimax_lojas_STO3_03_indv).
Corremos o cluster e as lojas individualmente, com as seguintes variáveis exógenas:
- Stock_inicial 
- Promo bin 1 (very low, low, moderate, high, very high)
- Outono, Verão e Primavera
- Feriados

Os resultados abaixo encontram-se [aqui](https://github.com/Gmarchi-silva/ISLA/blob/main/modeling/resultados_reais.xlsx). Através destes gráficos podemos ver que a previsão do modelo do cluster e das lojas são muito parecidas. 

![image](https://github.com/Gmarchi-silva/ISLA/assets/125706061/3d17a97e-d0e8-4d63-a36f-57f9b7cf55d4)

Nos gráficos abaixo é possível ver que o MAE, MSE e o RMSE do cluster e das lojas individuais andam próximos, no entanto parece-nos que o cluster tem um erro relativamente menor em comparação com as lojas individuais. No caso do MAPE varia de loja para loja sendo o modelo do cluster mais preciso numas lojas e os modelos individuais mais precisos noutras.

![image](https://github.com/Gmarchi-silva/ISLA/assets/125706061/88128b1c-2041-4ba4-bb10-7802efe9efa5)

Em relação ao AIC podemos ver que o modelo do cluster é muito melhor que os modelos individuais das lojas. Isto indica-nos que o modelo do cluster se ajusta melhor aos dados.

![image](https://github.com/Gmarchi-silva/ISLA/assets/125706061/eacac510-560d-4fa1-bbca-9570a7e967a1)


- Os resultados do Sarimax_cluster relativo a coeficientes, aic, bic,... encontram-se [aqui](https://github.com/Gmarchi-silva/ISLA/blob/main/modeling/Sarimax%20cluster/resultados_sarimax_cluster.csv)

- Os resultados do Sarimax_lojas_ST03_03_indv relativo a coeficientes, aic, bic,... encontram-se [aqui](https://github.com/Gmarchi-silva/ISLA/blob/main/modeling/Sarimax_lojas_STO3_03_indv/resultados_Sarimax_indv.csv)

## 5.	Conclusion

Através das variáveis exógenas foi possível retirar informações que permitem saber como estas influenciam a 'revenue'. A empresa a partir desta informação pode tentar manipular algumas das variáveis de forma a tentar aumentar a 'revenue'.

O modelo Sarimax para o cluster de 8 lojas permite fazer previsões mais acertadas em relação aos modelos individuais de cada loja. 

Num futuro projeto para este tipo de dados seria interessante utilizar um modelo de dados em painel.
