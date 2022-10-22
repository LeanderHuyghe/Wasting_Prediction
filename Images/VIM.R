library(VIM)
library(magrittr)

dataset <- read.csv("C:/Users/lonew/OneDrive/Email attachments/Documents/GitHub/Wasting_Prediction/data/semiyearly_chosen_columns.csv")
dataset <- subset(dataset,select=-c(X,MAM))
dataset <- subset(dataset,select=c(Price.of.water,n_conflict_total,phase3plus_perc_x,ndvi_score,next_prevalence,
 increase, increase_numeric))
aggr(dataset, col=c("blue","pink","orange"),sortVars=TRUE,labels=names(dataset),numbers=TRUE, cex.axis=.7, gap=3, ylab=c("Proportion of missing values","Distribution of Missing Values"))

imp_knn <- kNN(dataset)
aggr(imp_knn,delimiter = "_imp",, col=c("lightblue","pink","orange"),sortVars=TRUE,labels=names(dataset),numbers=TRUE, cex.axis=.7, gap=3, ylab=c("Proportion of missing values","Distribution of Missing Values"))

x <- dataset[, c("next_prevalence","Price.of.water","n_conflict_total","phase3plus_perc_x","ndvi_score")]
barMiss(x, only.miss = FALSE)


x_imp <- kNN(x)
# for missing values
# Red bands are missing values for x and blue dots are distrbituion of y
scattMiss(dataset[,c("phase3plus_perc_x","next_prevalence")],pch=16, main="Scatterplot of Price.of.water and next_prevalence")

imp_knn <- kNN(dataset, variable = "n_conflict_total")
dataset[, c("n_conflict_total", "Price.of.water")] %>%
  marginplot()

imp_knn[, c("n_conflict_total", "Price.of.water")] %>%
  marginplot(delimiter = "n_conflict_total_imp")




imp_knn <- kNN(dataset, variable = "n_conflict_total")
dataset[, c("n_conflict_total", "next_prevalence")] %>%
+   marginplot()
imp_knn[, c("NonD", "Span", "NonD_imp")] %>%
+   marginplot(delimiter = "_imp")
