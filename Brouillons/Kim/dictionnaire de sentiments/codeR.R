devtools::install_github("tidyverse/stringr")
library(quanteda) #la version 3.4 de R est nécessaire pour ce package
library(stringr)

dictfile <- tempfile()
download.file("http://dimension.usherbrooke.ca/voute/frlsd.zip", dictfile, mode = "wb")
unzip(dictfile, exdir = (td <- tempdir()))
dic_ton <- dictionary(file = paste(td, "frlsd.cat", sep = "/"))

#Mots négatifs
mots_neg <- unlist(dic_ton@.Data[1])
mots_neg <- mots_neg[-grep("\\*",mots_neg)]
mots_neg <- unique(unlist(str_split(mots_neg, pattern=" ")))
write.table(matrix(mots_neg, nrow=1), file ="mots_neg.csv", row.names=FALSE,col.names = FALSE,sep=";")

#Mots négatifs
mots_pos <- unlist(dic_ton@.Data[2])
mots_pos <- mots_pos[-grep("\\*",mots_pos)]
mots_pos <- unique(unlist(str_split(mots_pos, pattern=" ")))
write.table(matrix(mots_pos, nrow=1), file ="mots_pos.csv", row.names=FALSE,col.names = FALSE,sep=";")
