# Projeto II

## 1.	Enquadramento / Business Understanding

Foi disponibilizada informação de uma empresa cuja atividade é de comércio de roupa em lojas físicas presentes em diversas cidades da Turquia.
A empresa precisa de atualizar o armazenamento de roupa em cada loja para Outubro de 2019 por isso o objetivo principal passa por prever com precisão as vendas semanais de cada loja considerando  sazonalidade, tendência e variáveis explicativas complementando com uma análise dos dados que serão utilizados no modelo.
Neste sentido, e seguindo a metodologia CRISP-DM, pretendemos criar modelo (s) de previsão para as lojas e prever as vendas para Outubro de 2019 para cada loja, avaliando se a previsão é próxima do número de vendas real para que a gestão do armazenamento possa ser o mais eficiente possível e potenciar o aumento das vendas/receitas.

## 2.	Data Understanding

Os dados disponibilizados são em formato “csv” e correspondem a três tabelas (“sales”, “product” e “cities”).
Para a análise de dados utilizamos as ferramentas Excel, Python (Spyder e Google Colab) e Rstudio para extrair o melhor entendimento possível da informação presente nos dados.
Primeiro verificamos as variáveis presentes em cada uma das tabelas e o possível relacionamento existente entre elas. Neste caso, as tabelas “cities” e “product” estão relacionadas diretamente com a tabela “sales” pelas variáveis “store_id” e “product_id” respectivamente, podendo estas colunas serem consideradas chaves primárias visto que apenas contêm dados únicos (sendo chaves estrangeiras na tabela “sales”).

Para apoiar nesta análise recorremos à biblioteca “pandas_profiling” do Python (relatórios de cada tabela no repositório) e percebemos o seguinte:
  
  ### Tabela cities
  - A tabela “cities” contém 6 variáveis (1 numérica e 6 categóricas) e 63 observações, sem valores em falta:
  - Existem 63 lojas classificadas em 4 tipos e com 32 tamanhos entre elas sendo que nos tipos das lojas conseguimos perceber que são pequenas, médias e grandes, havendo uma especial pois é a única presente no tipo ST02.
  - As lojas são na Turquia e dispersas por 19 cidades diferentes, no entanto, 32 delas estão na cidade de Istanbul.

  ### Tabela Product
  - A tabela “product” contém 10 variáveis (3 numéricas e 7 categóricas) e 699 observações, com 100 valores em falta (1,4%)
- Existem registados 699 produtos diferentes segmentados em 10 clusters pela coluna “cluster_id” (havendo 50 produtos sem segmento e sendo o “cluster_0” o mais representativo com 450 produtos – 64,4%) que não conseguimos perceber com estes dados os critérios.
- As colunas “product_length”, “product_depth” e “product_width” caracterizam as dimensões do produto e possuem valores em falta (nem todos comuns às 3 colunas) e um registo zero.
- As 5 colunas “hierarchy…” classificam os produtos em vários níveis e nenhuma possui valores em falta.

  ### Tabela Sales
- A tabela “sales” contém 14 variáveis (6 numéricas e 8 categóricas) e 8.886.058 observações, com 35.271.795 valores em falta (28,4%), mas sem duplicados.
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
- Correlações:
- As ‘sales’ têm uma forte correlação positiva com a ‘revenue’, o que já seria de esperar
- A correlação de ‘sales’ com ‘stock’ é positiva mas pouco significativa
- O mesmo acontece com ‘sales’ e ‘price’ mas neste caso com correlação negativa
- A correlação entre ‘stock’ e ‘price’ é negativa mas não muito significativa
  
![Logo do GitHub](https://github.com/Gmarchi-silva/ISLA/blob/main/Pandas%20Profiling/Correla%C3%A7%C3%A3o.png)


## 3.	Data Preparation

 Para o tratamento e agregação dos dados todos que escolhemos foi utilizado o Python:
 
- Correção dos nomes  das cidades ("?" - substituir por "i") 
- Preencher os dados em falta nas colunas “promo_bin_1”, “promo_bin_2” e “promo_discount_type_2” com “NA” e na coluna “promo_discount_2” com “zero”.
- Agregação dos dados das 3 tabelas e eliminação das variáveis que não vamos utilizar: “city_id_old”, “Unnamed: 0”, “country_id”, “hierarchy1_id”, “hierarchy2_id”, “hierarchy3_id”, “hierarchy4_id”, “hierarchy5_id”, “product_length”, “product_depth”, “product_width”, “cluster_id”
- Verificação de nulos e eliminação das linhas que tinham nulos de “sales”, “revenue”, “stock”.
- Criação de uma coluna com o número contínuo de semanas com base na data.
- Eliminação da semana nº144 (última) por apenas ter dados de 1 dia

### Lojas anormais

Como temos lojas com valores de semana muito abaixo de 143 decidimos fazer uma análise particular e verificamos que algumas lojas destas têm o primeiro registo de vendas na semana 53 e depois têm um período sem vendas (sendo esse período diferente de loja para loja) o que nos leva a crer que o primeiro registo se tratou de um teste e que as semanas seguintes sem registos se deveram à preparação da loja para abertura definitiva e por isso eliminamos a semana 53 dessas lojas e utilizamos os dados apenas das semanas seguintes que tinham registos.
Existem também 2 lojas (S0007 e S0059) que começaram a vender em semanas diferentes pelo que utilizamos os dados apenas das semanas seguintes que tinham registos.
Temos ainda o caso especial da loja S0136 que percebemos que não está aberta todo o ano e, pela análise efetuada a loja terá fechado em setembro como habitualmente fez nos períodos homólogos de 2017 e 2018, pelo que a previsão para outubro de 2019 é que esteja fechada e por isso não haja vendas.
Depois deste tratamento dos dados agregamos tudo numa variável e “csv” final.



