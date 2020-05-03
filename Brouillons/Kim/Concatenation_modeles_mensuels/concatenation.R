setwd("C:/Users/Kim Antunez/Documents/Projets_autres/StatApp/data/CONCAT")

fichiers <- list.files("C:/Users/Kim Antunez/Documents/Projets_autres/StatApp/data/CONCAT")

lire <- function(x){
    return(read.table(x,sep=";",header=TRUE,stringsAsFactors = FALSE))
}

toto <- lapply(fichiers, lire)

tata <-rbind(toto[[1]],toto[[2]],toto[[3]],toto[[4]],toto[[5]],toto[[6]],toto[[7]],toto[[8]],toto[[9]],toto[[10]],toto[[11]])
tata[,2] <- NULL
tata$X <- as.Date(paste0(tata$X,"-01"))
ordre <- order(tata$X)
tata <- tata[ordre,]
tata$X <- substr(as.character(tata$X),1,7)

write.table(tata,"tata.csv",sep=";",col.names=TRUE,row.names=FALSE)

?write.table
#titi <- read.table("C:/Users/Kim Antunez/Documents/Projets_autres/StatApp/data/sentimental_analysis/Sentiment_brut_modele.csv",sep=";",header=TRUE)


