library(RJDemetra)
library(rJava)
test_jd2r<-function(s){
    if (is.null(s))
        return(NULL)
    desc<-.jfield(s, "S", "description")
    val<-.jfield(s, "D", "value")
    pval<-.jfield(s, "D", "pvalue")
    all<-c(val, pval)
    attr(all, "description")<-desc
    return (all)
}

jd2_seasonality_KruskalWallis<-function(series){
    js<-RJDemetra:::ts_r2jd(series)
    jtest<-.jcall("ec/tstoolkit/jdr/tests/SeasonalityTests", "Lec/tstoolkit/information/StatisticalTest;", "kruskalWallisTest", js)
    if (is.jnull(jtest))
        return (NULL)
    else{
        return (test_jd2r(jtest))
    }
}

jd2_seasonality_Friedman<-function(series){
    js<-RJDemetra:::ts_r2jd(series)
    jtest<-.jcall("ec/tstoolkit/jdr/tests/SeasonalityTests", "Lec/tstoolkit/information/StatisticalTest;", "friedmanTest", js)
    if (is.jnull(jtest))
        return (NULL)
    else{
        return (test_jd2r(jtest))
    }
}


jd2_seasonality_FTest<-function(series, ar=TRUE, nyears=0){
    js<-RJDemetra:::ts_r2jd(series)
    jtest<-.jcall("ec/tstoolkit/jdr/tests/SeasonalityTests", "Lec/tstoolkit/information/StatisticalTest;", "ftest", js, ar, as.integer(nyears))
    if (is.jnull(jtest))
        return (NULL)
    else{
        return (test_jd2r(jtest))
    }
}

jd2_seasonality_QSTest<-function(series, nyears=0, diff = -1, mean=TRUE){
    js<-RJDemetra:::ts_r2jd(series)
    jtest<-.jcall("ec/tstoolkit/jdr/tests/SeasonalityTests", "Lec/tstoolkit/information/StatisticalTest;", "qstest", js, as.integer(nyears), as.integer(diff), mean)
    if (is.null(jtest))
        return (NULL)
    else{
        return (test_jd2r(jtest))
    }
}

jd2_td_FTest<-function(series, ar=TRUE, nyears=0){
    js<-RJDemetra:::ts_r2jd(series)
    jtest<-.jcall("ec/tstoolkit/jdr/tests/TradingDaysTests", "Lec/tstoolkit/information/StatisticalTest;", "ftest", js, ar, as.integer(nyears))
    if (is.jnull(jtest))
        return (NULL)
    else{
        return (test_jd2r(jtest))
    }
}

# jd2_seasonality_Friedman(modeles[,ind_retenu]) # test s'il y a une saisonnalité stable
# jd2_seasonality_KruskalWallis(modeles[,ind_retenu]) # pas de saisonnalité
# jd2_seasonality_FTest(modeles[,ind_retenu]) # présence de saisonnalité
# jd2_seasonality_QSTest(modeles[,ind_retenu])
# jd2_td_FTest(modeles[,ind_retenu]) # pas d'effet JO
# 
# 
# jd2_seasonality_Friedman(ipi_c_eu[,"FR"]) # test s'il y a une saisonnalité
# jd2_seasonality_KruskalWallis(ipi_c_eu[,"FR"])
# jd2_seasonality_FTest(ipi_c_eu[,"FR"])
# jd2_seasonality_QSTest(ipi_c_eu[,"FR"])
# jd2_td_FTest(ipi_c_eu[,"FR"]) # pas d'effet jour ouvrable
# 
# 
# mod <- x13(ts(scale(modeles[,ind_retenu]),start = start(modeles[,ind_retenu]), frequency = 12),
#            "RSA4c")
# plot(mod$decomposition)
# 
# plot(mod$final, type = "sa-"
# get_indicators(mod,"sa")[[1]]
# jd2_seasonality_Friedman(get_indicators(mod,"sa")[[1]]) # test s'il y a une saisonnalité
# jd2_seasonality_KruskalWallis(get_indicators(mod,"sa")[[1]])
# jd2_seasonality_FTest(get_indicators(mod,"sa")[[1]])
# jd2_seasonality_QSTest(get_indicators(mod,"sa")[[1]])
# jd2_td_FTest(get_indicators(mod,"sa")[[1]])


