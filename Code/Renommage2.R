setwd("data/ens/gensim")

nouveau_nom <- sapply(strsplit(list.files(pattern="^dim"), "_"),function(x){
    paste(c(grep("^dim",x, value = TRUE),
            grep("^ep",x, value = TRUE),
            grep("^w",x, value = TRUE),
            grep("^lr",x, value = TRUE),
            grep("^seed",x, value = TRUE)),
          collapse =  "_")
})

#vérifier
data.frame(list.files(pattern="^dim"),nouveau_nom) 

#renommer!
file.rename(list.files(pattern="^dim"),nouveau_nom)
