rm(list=ls())

setwd('C:/Users/Kim Antunez/Documents/Projets_autres/StatApp/Brouillons/Kim/Base barometre')


#base_entiere <- readRDS("C:/Users/Kim Antunez/Documents/DREES/R Shiny Application/Appli version en ligne/data/bdd_2017_2018.rds")
#attr(base_entiere$og1,"label")

in1 <- "IN1. La société française aujourd'hui, vous paraît-elle plutôt juste ou plutôt injuste ?"
in2 <- "IN2. Globalement, depuis 5 ans, diriez-vous que les inégalités en France…"
in3 <- "IN3. À l'avenir, pensez-vous que les inégalités en France…"
og1 <-  "OG1. Vous personnellement, comment qualifieriez-vous votre situation actuelle ?"
og2_ab <- "OG2_AB. Comparée à votre situation actuelle, diriez-vous de la situation de vos parents, au même âge, qu'elle était... ?"
og3_1 <- "OG3_1. Quand vous pensez à l'avenir, êtes-vous plutôt optimiste ou plutôt pessimiste... pour vous-même ?"
og3_2 <-  "OG3_2. Quand vous pensez à l'avenir, êtes-vous plutôt optimiste ou plutôt pessimiste... pour vos enfants ou les générations futures ?"
og4_3 <- "OG4_3. Êtes-vous personnellement préoccupé(e) par… la pauvreté ?"
og4_6 <- "OG4_6. Êtes-vous personnellement  préoccupé(e) par… le chômage ?"
pe2 <- "PE2. À l'avenir, pensez-vous que la pauvreté et l'exclusion en France…"
pe1 <- "PE1. Selon vous, depuis 5 ans, la pauvreté et l'exclusion en France..."

library(dplyr)

mergeur <- data.frame(Annee=2011:2018)

bdd_in1 <- read.csv("data/Barometre-opinion-DREES-in1.csv",encoding = "UTF-8",stringsAsFactors = FALSE) %>%
    filter(Modalités!="(NSP)") %>% 
    mutate(indicateur=ifelse(Modalités=="Plutôt juste",1,0)) %>% 
    group_by(Année,indicateur) %>% 
    summarise(denom=sum(Effectif)) %>% 
    mutate(num=denom*indicateur) %>% 
    group_by(Année) %>% 
    summarise(denom=sum(denom),num=sum(num)) %>% 
    mutate(indic=100*num/denom) %>% 
    select(Année,indic) %>% 
    right_join(mergeur,by=c("Année"="Annee"))

bdd_in2 <- read.csv("data/Barometre-opinion-DREES-in2.csv",encoding = "UTF-8",stringsAsFactors = FALSE) %>%
    filter(Modalités!="(NSP)") %>% 
    mutate(indicateur=ifelse(Modalités%in%c("Ont plutôt diminué","(Sont restées stables)"),1,0)) %>% 
    group_by(Année,indicateur) %>% 
    summarise(denom=sum(Effectif)) %>% 
    mutate(num=denom*indicateur) %>% 
    group_by(Année) %>% 
    summarise(denom=sum(denom),num=sum(num)) %>% 
    mutate(indic=100*num/denom) %>% 
    select(Année,indic) %>% 
    right_join(mergeur,by=c("Année"="Annee"))

bdd_in3 <- read.csv("data/Barometre-opinion-DREES-in3.csv",encoding = "UTF-8",stringsAsFactors = FALSE) %>%
    filter(Modalités!="(NSP)") %>% 
    mutate(indicateur=ifelse(Modalités%in%c("Vont plutôt diminuer","(Resteront stables)"),1,0)) %>% 
    group_by(Année,indicateur) %>% 
    summarise(denom=sum(Effectif)) %>% 
    mutate(num=denom*indicateur) %>% 
    group_by(Année) %>% 
    summarise(denom=sum(denom),num=sum(num)) %>% 
    mutate(indic=100*num/denom) %>% 
    select(Année,indic) %>% 
    right_join(mergeur,by=c("Année"="Annee"))


bdd_og1 <- read.csv("data/Barometre-opinion-DREES-og1.csv",encoding = "UTF-8",stringsAsFactors = FALSE) %>%
    filter(Modalités!="(NSP)") %>% 
    mutate(indicateur=ifelse(Modalités%in%c("Très bonne","Assez bonne"),1,0)) %>% 
    group_by(Année,indicateur) %>% 
    summarise(denom=sum(Effectif)) %>% 
    mutate(num=denom*indicateur) %>% 
    group_by(Année) %>% 
    summarise(denom=sum(denom),num=sum(num)) %>% 
    mutate(indic=100*num/denom) %>% 
    select(Année,indic) %>% 
    right_join(mergeur,by=c("Année"="Annee"))


bdd_og2_ab <- read.csv("data/Barometre-opinion-DREES-og2_ab.csv",encoding = "UTF-8",stringsAsFactors = FALSE) %>%
    filter(Modalités!="(NSP)") %>% 
    mutate(indicateur=ifelse(Modalités%in%c("Bien meilleure", "Plutôt meilleure","A peu près identique"),1,0)) %>% 
    group_by(Année,indicateur) %>% 
    summarise(denom=sum(Effectif)) %>% 
    mutate(num=denom*indicateur) %>% 
    group_by(Année) %>% 
    summarise(denom=sum(denom),num=sum(num)) %>% 
    mutate(indic=100*num/denom) %>% 
    select(Année,indic) %>% 
    right_join(mergeur,by=c("Année"="Annee"))


bdd_og3_1 <- read.csv("data/Barometre-opinion-DREES-og3_1.csv",encoding = "UTF-8",stringsAsFactors = FALSE) %>%
    filter(Modalités!="(NSP)") %>% 
    mutate(indicateur=ifelse(Modalités%in%c("Très optimiste", "Plutôt optimiste","A peu près identique"),1,0)) %>% 
    group_by(Année,indicateur) %>% 
    summarise(denom=sum(Effectif)) %>% 
    mutate(num=denom*indicateur) %>% 
    group_by(Année) %>% 
    summarise(denom=sum(denom),num=sum(num)) %>% 
    mutate(indic=100*num/denom) %>% 
    select(Année,indic) %>% 
    right_join(mergeur,by=c("Année"="Annee"))


bdd_og3_2 <- read.csv("data/Barometre-opinion-DREES-og3_2.csv",encoding = "UTF-8",stringsAsFactors = FALSE) %>%
    filter(Modalités!="(NSP)") %>% 
    mutate(indicateur=ifelse(Modalités%in%c("Très optimiste", "Plutôt optimiste","A peu près identique"),1,0)) %>% 
    group_by(Année,indicateur) %>% 
    summarise(denom=sum(Effectif)) %>% 
    mutate(num=denom*indicateur) %>% 
    group_by(Année) %>% 
    summarise(denom=sum(denom),num=sum(num)) %>% 
    mutate(indic=100*num/denom) %>% 
    select(Année,indic) %>% 
    right_join(mergeur,by=c("Année"="Annee"))



bdd_og4_3 <- read.csv("data/Barometre-opinion-DREES-og4_3.csv",encoding = "UTF-8",stringsAsFactors = FALSE) %>%
    filter(Modalités!="(NSP)") %>% 
    mutate(indicateur=ifelse(Modalités%in%c("Peu","Pas du tout"),1,0)) %>% 
    group_by(Année,indicateur) %>% 
    summarise(denom=sum(Effectif)) %>% 
    mutate(num=denom*indicateur) %>% 
    group_by(Année) %>% 
    summarise(denom=sum(denom),num=sum(num)) %>% 
    mutate(indic=100*num/denom) %>% 
    select(Année,indic)  %>% 
    right_join(mergeur,by=c("Année"="Annee"))


bdd_og4_6 <- read.csv("data/Barometre-opinion-DREES-og4_6.csv",encoding = "UTF-8",stringsAsFactors = FALSE) %>%
    filter(Modalités!="(NSP)") %>% 
    mutate(indicateur=ifelse(Modalités%in%c("Peu","Pas du tout"),1,0)) %>% 
    group_by(Année,indicateur) %>% 
    summarise(denom=sum(Effectif)) %>% 
    mutate(num=denom*indicateur) %>% 
    group_by(Année) %>% 
    summarise(denom=sum(denom),num=sum(num)) %>% 
    mutate(indic=100*num/denom) %>% 
    select(Année,indic) %>% 
    right_join(mergeur,by=c("Année"="Annee"))


bdd_pe1 <- read.csv("data/Barometre-opinion-DREES-pe1.csv",encoding = "UTF-8",stringsAsFactors = FALSE) %>%
    filter(Modalités!="(NSP)") %>% 
    mutate(indicateur=ifelse(Modalités%in%c("(Sont restées stables)","Ont diminué"),1,0)) %>% 
    group_by(Année,indicateur) %>% 
    summarise(denom=sum(Effectif)) %>% 
    mutate(num=denom*indicateur) %>% 
    group_by(Année) %>% 
    summarise(denom=sum(denom),num=sum(num)) %>% 
    mutate(indic=100*num/denom) %>% 
    select(Année,indic) %>% 
    right_join(mergeur,by=c("Année"="Annee"))



bdd_pe2 <- read.csv("data/Barometre-opinion-DREES-pe2.csv",encoding = "UTF-8",stringsAsFactors = FALSE) %>%
    filter(Modalités!="(NSP)") %>% 
    mutate(indicateur=ifelse(Modalités%in%c("(Resteront stables)","Vont plutôt diminuer"),1,0)) %>% 
    group_by(Année,indicateur) %>% 
    summarise(denom=sum(Effectif)) %>% 
    mutate(num=denom*indicateur) %>% 
    group_by(Année) %>% 
    summarise(denom=sum(denom),num=sum(num)) %>% 
    mutate(indic=100*num/denom) %>% 
    select(Année,indic) %>% 
    right_join(mergeur,by=c("Année"="Annee"))

base_finale <- bind_cols(bdd_in1, bdd_in2, bdd_in3, bdd_og1, bdd_og2_ab, bdd_og3_1, bdd_og3_2 , bdd_og4_3, bdd_og4_6, bdd_pe1, bdd_pe2)
base_finale <- base_finale[,c(1,seq(2,22,2))]
colnames(base_finale) <- c("Annee","in1", "in2", "in3", "og1", "og2_ab", "og3_1", "og3_2", "og4_3", "og4_6", "pe1", "pe2")


attr(base_finale$in1,"label") <- in1
attr(base_finale$in2,"label") <- in2
attr(base_finale$in3,"label") <- in3
attr(base_finale$og1,"label") <- og1
attr(base_finale$og2_ab,"label") <- og2_ab
attr(base_finale$og3_1,"label") <- og3_1
attr(base_finale$og4_3,"label") <- og4_3
attr(base_finale$og4_6,"label") <- og4_6
attr(base_finale$pe1,"label") <- pe1
attr(base_finale$pe2,"label") <- pe2

saveRDS(base_finale,"barometre.RDS")
