
library(dygraphs)
library(lawstat)

data2<-read.csv("C:/Users/jpaul/OneDrive/Ambiente de Trabalho/Projeto2/Loja_Sales_Revenue.csv")
data3<-read.csv("C:/Users/jpaul/OneDrive/Ambiente de Trabalho/Projeto2/Loja_Sales_Revenue2.csv")

loja2<-data3[which(data3$store_id == 'S0002'),]
loja2_0<-loja2[,c(2,4:5)]
dygraph(loja2_0)
#Avaliar os picos entre lojas S0007 S0045  e outras

seried <- diff(loja2[,4])
#Pelo gráfico abaixo parece ser homocedastica, a variabilidade não varia ao longo do tempo 
plot(ts(seried))

#Para confirmar a linha acima faz-se o teste de Levene. Se p-value > 1% não se rejeita homocedasticidade
grupos <- rep(1:5, each = 29) [-c(144:145)]
levene.test(seried, group = grupos)

#
monthplot(loja2[,5],labels = 1L:52L,times = loja2$week_number)

#contar número de dados por semana
list<-unique(data$store_id)
mat0<-matrix(data=0, nrow=length(list),ncol=2)
for(i in 1:length(list)){
  mat0[i,]=t(c(list[i],length(which(data2$store_id == list[i]))))
}


