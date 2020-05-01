rm(list=ls())

#setwd('C:/Users/torna/Documents/StatApp/StatApp')
setwd('C:/Users/Kim Antunez/Documents/Projets_autres/StatApp')
#setwd('/Users/alainquartierlatente/Desktop/Ensae/StatApp')
#setwd('/home/aqlt/Documents/Ensae/StatApp')

devtools::install_github("aqlt/AQLTools")
library(AQLTools)
library(ggplot2)
library(dplyr)
library(corrplot)

#################################
##### Chargement des données #####
##################################

#### Indices tweets
tweets <- read.table("data/sentimental_analysis/Sentiment_brut_modele.csv", sep = ";", stringsAsFactors = FALSE, header=TRUE)
colnames(tweets)[1] <- "Date"

tweets_an <- tweets %>% 
    mutate(Annee=as.numeric(substr(Date,1,4))) %>% 
    group_by(Annee) %>% 
    summarise_if(is.numeric, mean, na.rm = TRUE)

tweets_norm <- cbind(tweets[,1],
                     as.data.frame(apply(tweets[,-1],2,function(x){100 * x / x[1]})))
colnames(tweets_norm)[1] <- "Date"

tweets_an_norm <- cbind(tweets_an[,1],
                     apply(tweets_an[,-1],2,function(x){100 * x / x[1]}))

#tweets <- ymd_ts(tweets, sep_date = "-")
#tweets_norm <- ymd_ts(tweets_norm, sep_date = "-")
#tweets_an <- ts(tweets_an[,-1], start=min(tweets_an$Annee))
#tweets_an_norm <- ts(tweets_an_norm[,-1], start=min(tweets_an_norm$Annee))


#### Données CAMME
camme <- read.table("data/sentimental_analysis/camme.csv", sep = ";", stringsAsFactors = FALSE, header=TRUE)

camme_norm <- camme
camme_norm$Indice <- 100 * camme_norm[,-1] / camme_norm[1,-1]
colnames(camme_norm) <- c("Date","Camme")

#camme <- ymd_ts(camme, sep_date = "-")
#camme_norm <- ymd_ts(camme_norm, sep_date = "-")

#### Données Baromètre
baro <- readRDS("data/barometre.RDS")

baro_norm <- cbind(baro[,1],
                     as.data.frame(apply(baro[,-1],2,function(x){100 * x / x[1]})))

#baro <- ts(baro[,-1], start=min(baro$Annee))
#baro_norm <- ts(baro_norm[,-1], start=min(baro_norm$Annee))

##################################
##### Données mensuelles #####
##################################

indices_mois <- merge(tweets_norm, camme_norm, by="Date",stringAsFactors=FALSE)

corrplot(cor(indices_mois[,-1]), type="upper", order="hclust", tl.col="black", tl.srt=45)

graph_ts(ymd_ts(indices_mois, sep_date = "-"))


#### en évol

indices_mois_evol <- cbind(indices_mois[,1],
                           as.data.frame(apply(indices_mois[,-1],2,function(x){100 * (x - c(NA,x[-length(x)]))/ c(NA,x[-length(x)]) })))


corrplot(cor(indices_mois_evol[-1,-1]), type="upper", order="hclust", tl.col="black", tl.srt=45)


graph_ts(ymd_ts(indices_mois_evol[,c(1,5,8)]))
# a decaler ?

##################################
##### Données annuelles #####
##################################

indices_an <- merge(tweets_an_norm, baro_norm, by="Annee",stringAsFactors=FALSE)

cor(indices_an[,-c(1)])

corrplot(cor(indices_an[,-c(1,2,3,4,6,7,12)]), type="upper", order="hclust", tl.col="black", tl.srt=45)





library(AQLTools)




AQLTools::hc_stocks(ymd_ts(indices_an, sep_date = "-"))


############## brouillon


serie = ts.union(tweets_an_norm,baro_norm)
graph_ts(serie[,1:10])

?graph_ts

plot(serie[,1:10],plot.type = "single")

?plot.ts

library(dygraphs)
dygraph(serie)


