library(tidyverse)
library(patchwork)
library(caret)
library(vcd)
library(gridExtra)
library(knitr)
library(corrplot)
library(scales)
library(lme4)
library(InformationValue)
library(ROCR)
library(rpart)
library(randomForest)
library(xgboost)
library(MASS)
library(ggmosaic)
library(e1071)
library(ranger)
library(penalized)
library(rpart.plot)
library(ggcorrplot)
library(caTools)
library(doMC)
registerDoMC(cores=4)
bankChurn <- read_csv('Churn_Modelling.csv')

glimpse(bankChurn)
bankChurn <- bankChurn %>% 
  dplyr::select(-RowNumber, -CustomerId, -Surname) %>%
  mutate(Geography = as.factor(Geography),
         Gender = as.factor(Gender),
         HasCrCard = as.factor(HasCrCard),
         IsActiveMember = as.factor(IsActiveMember),
         Exited = as.factor(Exited),
         Tenure = as.factor(Tenure),
         NumOfProducts = as.factor(NumOfProducts))
sapply(bankChurn, function(x) sum(is.na(x)))
summary(bankChurn)

ggplot(bankChurn, aes(Exited, fill = Exited)) +
  geom_bar() +
  theme(legend.position = 'none')
table(bankChurn$Exited)
round(prop.table(table(bankChurn$Exited)),3)

bankChurn %>%
  keep(is.numeric) %>%
  gather() %>%
  ggplot() +
  geom_histogram(mapping = aes(x=value,fill=key), color="black") +
  facet_wrap(~ key, scales = "free") +
  theme_minimal() +
  theme(legend.position = 'none')

numericVarName <- names(which(sapply(bankChurn, is.numeric)))
corr <- cor(bankChurn[,numericVarName], use = 'pairwise.complete.obs')
ggcorrplot(corr, lab = TRUE)

bankChurn %>%
  dplyr::select(-Exited) %>% 
  keep(is.factor) %>%
  gather() %>%
  group_by(key, value) %>% 
  summarize(n = n()) %>% 
  ggplot() +
  geom_bar(mapping=aes(x = value, y = n, fill=key), color="black", stat='identity') + 
  coord_flip() +
  facet_wrap(~ key, scales = "free") +
  theme_minimal() +
  theme(legend.position = 'none')
#age
age_hist <- ggplot(bankChurn, aes(x = Age, fill = Exited)) +
  geom_histogram(binwidth = 5) +
  theme_minimal() +
  scale_x_continuous(breaks = seq(0,100,by=10), labels = comma)

age_boxplot <- ggplot(bankChurn, aes(x = Exited, y = Age, fill = Exited)) +
  geom_boxplot() + 
  theme_minimal() +
  theme(legend.position = 'none')

age_hist | age_boxplot

balance_hist <- ggplot(bankChurn, aes(x = Balance, fill = Exited)) +
  geom_histogram() +
  theme_minimal() +
  scale_x_continuous(breaks = seq(0,255000,by=30000), labels = comma) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5))

balance_box <- ggplot(bankChurn, aes(x = Exited, y = Balance, fill = Exited)) +
  geom_boxplot() + 
  theme_minimal() +
  theme(legend.position = 'none')

balance_hist | balance_box
chi.square <- vector()
p.value <- vector()
cateVar <- bankChurn %>% 
  dplyr::select(-Exited) %>% 
  keep(is.factor)

for (i in 1:length(cateVar)) {
  p.value[i] <- chisq.test(bankChurn$Exited, unname(unlist(cateVar[i])), correct = FALSE)[3]$p.value
  chi.square[i] <- unname(chisq.test(bankChurn$Exited, unname(unlist(cateVar[i])), correct = FALSE)[1]$statistic)
}

chi_sqaure_test <- tibble(variable = names(cateVar)) %>% 
  add_column(chi.square = chi.square) %>% 
  add_column(p.value = p.value)
knitr::kable(chi_sqaure_test)

credit_hist <- ggplot(bankChurn, aes(x = CreditScore, fill = Exited)) +
  geom_histogram() +
  theme_minimal() +
  #scale_x_continuous(breaks = seq(0,255000,by=30000), labels = comma) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5))

credit_box <- ggplot(bankChurn, aes(x = Exited, y = CreditScore, fill = Exited)) +
  geom_boxplot() + 
  theme_minimal() +
  theme(legend.position = 'none')

credit_hist | credit_box
bankChurn <- bankChurn %>% 
  dplyr::select(-Tenure, -HasCrCard)

set.seed(1234)
sample_set <- bankChurn %>%
  pull(.) %>% 
  sample.split(SplitRatio = .7)

bankTrain <- subset(bankChurn, sample_set == TRUE)
bankTest <- subset(bankChurn, sample_set == FALSE)

round(prop.table(table(bankChurn$Exited)),3)
round(prop.table(table(bankTrain$Exited)),3)
round(prop.table(table(bankTest$Exited)),3)

#Logistic Regression
logit.mod <- glm(Exited ~.,
            family = binomial(link = 'logit'), 
            data = bankTrain)
summary(logit.mod)
logit.pred.prob <- predict(logit.mod, 
                    bankTest, type = 'response')
logit.pred <- as.factor(ifelse(logit.pred.prob
                        > 0.5, 1, 0))
head(bankTest,10)
head(logit.pred.prob,10)
caret::confusionMatrix(logit.pred, 
        bankTest$Exited, positive = "1")

#Decision Tree
ctrl <-
  trainControl(method = "cv", #cross-validation
               number = 10, #10-fold
               selectionFunction = "best")

grid <- 
  expand.grid(
    .cp = seq(from=0.0001, to=0.005, by=0.0001)
  )
set.seed(1234)
tree.mod <-
  train(
    Exited ~.,
    data = bankTrain,
    method = "rpart",
    metric = "Kappa",
    trControl = ctrl,
    tuneGrid = grid
  )

tree.mod

tree.pred.prob <- predict(tree.mod, bankTest, type = "prob")
tree.pred <- predict(tree.mod, bankTest, type = "raw")
caret::confusionMatrix(tree.pred, bankTest$Exited, positive = "1")
