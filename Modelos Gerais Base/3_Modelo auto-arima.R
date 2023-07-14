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
df_semana.ts <- ts(df_semana, frequency = 52)
class(df_semana.ts)
print(df_semana.ts)
plot.ts(df_semana.ts)

# Decomposição em componentes (sazonalidade, tendência e ciclo)-----------------
components <- decompose(df_semana.ts)
plot(components)

plot(components$x, main = "Observed data")
plot(components$trend, main = "Trend")
plot(components$seasonal, main = "Seasonal")
plot(components$random, main = "Random")

# Criação do modelo com base em 'sales' ----------------------------------------
fit.arima.auto.sales <- auto.arima(df_semana$sales)
summary(fit.arima.auto.sales)
checkresiduals(fit.arima.auto.sales)

# Criação do modelo com base em 'revenue' --------------------------------------
fit.arima.auto.revenue <- auto.arima(df_semana$revenue)
summary(fit.arima.auto.revenue)
checkresiduals(fit.arima.auto.revenue)

