rm(list = ls())
graphics.off()

# Bibliotecas ------------------------------------------------------------------
library("dplyr")
#install.packages("tidyverse")
if (!require("forecast")) install.packages("forecast")
library(forecast) # Forecasting Functions for Time Series

# Leitura dos dados ------------------------------------------------------------

sales <- read.csv("C:/Users/igort/Documents/Igor/ISLA GAIA/14.Projeto II/Drive/Raw_table.csv")

dim(sales)
head(sales)
tail(sales)
str(sales)

# Preparação dos dados ---------------------------------------------------------
df <- data.frame(sales)
df <- df |> select(date, store_id, sales, revenue)
str(df)
df$date <- as.Date(df$date)
df <- subset(df, date < as.Date("2019-10-01"))
tail(df)

# Visualização gráfica temporal dos dados --------------------------------------
plot(df$sales, type = "o", main = "Weekly Sales")
axis(1, at = seq.Date(min(df$date), max(df$date), "week"), labels = NA)

plot(df$revenue, type = "o", main = "Weekly Revenue")
axis(1, at = seq.Date(min(df$revenue), max(df$date), "week"), labels = NA)


plot(df$sales, main = "Weekly Sales - lm")
abline(reg = lm(sales ~ date, data = df), lty = 2, lwd = 2, col = 2)

plot(df$revenue, main = "Weekly Revenue - lm")
abline(reg = lm(revenue ~ date, data = df), lty = 2, lwd = 2, col = 2)

# Agregação dos dados por semana -----------------------------------------------
df_semana <- aggregate(cbind(sales, revenue) ~ format(date, "%Y-%U"), data = df, FUN = sum)
df_semana <- setNames(df_semana, c("year-week", "sales", "revenue"))
df_semana
str(df_semana)


# Converter em série temporal --------------------------------------------------
df_semana.sales.ts <- ts(df_semana$sales, frequency = 52)
df_semana.revenue.ts <- ts(df_semana$revenue, frequency = 52)


plot.ts(df_semana.sales.ts)
str(df_semana.sales.ts)

plot.ts(df_semana.revenue.ts)
str(df_semana.revenue.ts)

# Decomposição em componentes (sazonalidade, tendência e ciclo)-----------------
components.sales <- decompose(df_semana.sales.ts)
plot(components.sales)

components.revenue <- decompose(df_semana.revenue.ts)
plot(components.revenue)

plot(components.sales$x, main = "Observed data 'sales'")
plot(components.sales$trend, main = "Trend 'sales'")
plot(components.sales$seasonal, main = "Seasonal 'sales'")
plot(components.sales$random, main = "Random 'sales'")

plot(components.revenue$x, main = "Observed data 'revenue'")
plot(components.revenue$trend, main = "Trend" 'revenue')
plot(components.revenue$seasonal, main = "Seasonal" 'revenue')
plot(components.revenue$random, main = "Random" 'revenue')

# Definição dos parâmetros ARIMA para 'sales' ---------------------------------

# Valor para o parâmetro d(I) e D ----------------------------------------------
# Visualização da diferença em 'sales'
plot(x = df$date[-(1:13)],
     y = diff(df$sales, lag = 13), # lag = 13
     type = "l",
     xlab = "Weeks",
     ylab = "Sales",
     main = "Difference in Sales (lag=13)")

d <- 0 # dado que a série já parece ser estacionária (mas tentar com 1 também)
D <- 0

# Valor para o parâmetro q(MA) -------------------------------------------------

# Gráfico auto-correlação
forecast::Acf(df$sales)

q <- 3 # primeiro lag de dimensão mais considerável acima do nível de confiança
Q <- 3

# Valor para o parâmetro p(AR) -------------------------------------------------

# Gráfico auto-correlação parcial
forecast::Acf(df$sales, type = "partial")

p <- 3 # primeiro lag de dimensão mais considerável acima do nível de confiança
P <- 3

# Criação manual do modelo com base em 'sales' ---------------------------------
fit.arima.sales <- Arima(df_semana.sales.ts, order = c(p, d, q))
summary(fit.arima.sales)
checkresiduals(fit.arima.sales)

# Definição dos parâmetros ARIMA para 'revenue' ---------------------------------

# Valor para o parâmetro d(I) e D ----------------------------------------------
# Visualização da diferença em 'sales'
plot(x = df$date[-1],
     y = diff(df$revenue, lag = 1), # lag = 1
     type = "l",
     xlab = "Weeks",
     ylab = "Revenue",
     main = "Difference in Revenue (lag=1)")

d <- 1 
D <- 1

# Valor para o parâmetro q(MA) -------------------------------------------------

# Gráfico auto-correlação
forecast::Acf(df$revenue)

q <- 3 # primeiro lag de dimensão mais considerável acima do nível de confiança
Q <- 3

# Valor para o parâmetro p(AR) -------------------------------------------------

# Gráfico auto-correlação parcial
forecast::Acf(df$revenue, type = "partial")

p <- 3 # primeiro lag de dimensão mais considerável acima do nível de confiança
P <- 3

# Criação manual do modelo com base em 'revenue' -------------------------------
fit.arima.revenue <- Arima(df_semana.revenue.ts, order = c(p, d, q))
summary(fit.arima.revenue)
checkresiduals(fit.arima.revenue)


# Comparação entre modelos -----------------------------------------------------

# O melhor modelo pode ser escolhido pelo menor AIC ou BIC
#   AIC: Akaike information criterion
#   BIC: Bayesian information criterion
data.frame(AIC = round(c(fit.arima.sales$aic, fit.arima.revenue$aic), 1),
           BIC = round(c(fit.arima.sales$bic, fit.arima.revenue$bic), 1),
           row.names = c("sales", "revenue"))

# Previsões dos modelos para 5 semanas -----------------------------------------

# Previsão 'sales'
predict.arima.sales <- forecast(fit.arima.sales, h = 5)
predict.arima.sales
plot(predict.arima.sales,
     sub = "manual setting of parameters",
     col = "navyblue",
     lwd = 2)


# Previsão 'revenue'
predict.arima.revenue <- forecast(fit.arima.revenue, h = 5)
predict.arima.revenue
plot(predict.arima.revenue,
     sub = "manual setting of parameters",
     col = "navyblue",
     lwd = 2)
