setwd("C:/Users/torna/Documents/StatApp/StatApp/data/sentimental_analysis/")

x <- readLines("French-SA/tweets-test.csv",encoding="UTF-8")[-1]
tweets <- gsub("^\\d*,","",x)
polarity <- as.numeric(substr(x,0,1))
table(polarity)
tweets <- tweets[-which(is.na(polarity))]
polarity <- polarity[-which(is.na(polarity))]
table(polarity)
data <- data.frame(tweets = tweets, polarity = polarity, stringsAsFactors = FALSE)
data_final <- data[data$tweets!="",]
write.table(data_final, file = "bdd-test.csv", quote = F,row.names = FALSE, fileEncoding = "UTF-8",
            sep = ";/;")


x <- readLines("French-SA/tweets-train.csv",encoding="UTF-8")[-1]
tweets <- gsub("^\\d*,","",x)
polarity <- as.numeric(substr(x,0,1))
polarity[!polarity%in%c(0,4)] <- NA
tweets <- tweets[-which(is.na(polarity))]
polarity <- polarity[-which(is.na(polarity))]
table(polarity)
data <- data.frame(tweets = tweets, polarity = polarity, stringsAsFactors = FALSE)
data_final <- data[!(data$tweets %in% c(""," ","n/a","??? ????!")),]
write.table(data_final, file = "bdd-train.csv", quote = F,row.names = FALSE, fileEncoding = "UTF-8",
            sep = ";/;")
