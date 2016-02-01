setwd(wd)
data<-read.csv("pml-training.csv")
library("caret")
library("ggplot2")

# data slicing
set.seed(1222)
inBuild<-createDataPartition(data$classe, p=0.75, list=FALSE)
validation<-data[-inBuild,]
buildData<-data[inBuild,]
inTrain<-createDataPartition(buildData$classe, p=0.7, list=FALSE)
training<-buildData[inTrain,]
testing<-buildData[-inTrain,]

# classes of predictors (can't use apply funcion because it converts data frame into matrix)
classes<-factor (levels=c("integer", "character", "factor", "numeric"))
for (i in 1:ncol(training)) {
        classes[i]<-class(training[,i])
}

summary(classes)

# levels of factors
factors<-which(classes=="factor")
for (i in 3:length(factors)) {
        print(i)
        print(colnames(training)[factors[i]])
        print(levels(training[,factors[i]]))
}

# user_name  does not influence
# raw_timestamp_part_1 and raw_timestamp_part_2 are not factors
# there are only 2 real factors (classe and new_window)
# 9 variables don't contain any informantion (NA, 0)
# other 24 variables actually are numeric

# new_window (the only factor) doesn't seem to influence the class
prop.table(table(training$classe,training$new_window),1)

# change class of 24 variables
var_num<-c(4,5,7,8,10,11,13,14,15,16,17,18,19,20,22,23,25,26,28,29,31,32,34,35) # numbers of factors which are not factors
training[,factors[var_num]]<-apply(training[,factors[var_num]],2,as.numeric)
for (i in 1:ncol(training)) {
        classes[i]<-class(training[,i])
}

summary(classes) # to see the result of changes

# now all the factors do not influence the outcome
factors<-which(classes=="factor")
unused_var<-factors

# varibles with no variance
nsv<-nearZeroVar(training, saveMetrics=TRUE)
unused_var<-c(unused_var, which(nsv$nzv==TRUE))
unused_var<-unused_var[order(unused_var)]
unused_var<-unique(unused_var)

# variables with a lot of NA'S
num_int<-c(which(classes=="integer"), which(classes=="numeric"))
num_int<-num_int[order(num_int)]
num_int<-num_int[-(1:3)] # exclude number and timestamp
num_int<-num_int[-which(num_int%in%unused_var==TRUE)]

sum<-apply(training[,num_int], 2, summary)
NA_num<-numeric(length=length(sum))
for (i in 1:length(sum)){
        NA_num[i]<-sum[[i]][7]
}

unused_var<-c(unused_var, num_int[which(NA_num>10000)])
unused_var<-unused_var[order(unused_var)]
num_int<-num_int[-which(NA_num>10000)]

# plots with integers and numeric
plots<-list()
for (i in 1:length(num_int)){
        plots[[i]]=plot(training_out[,num_int[i]], training_out$classe, xlab=colnames(training_out)[num_int[i]], ylab="classe", main=i)
}

# exclude variables which don't have any influence
unused_var<-c(unused_var,num_int[c(6, 9:10, 12, 16:18, 20:22, 24:31, 37:41, 43:50, 52:53)])
unused_var<-unused_var[order(unused_var)]

# numbers of variables to be put into a model
var<-num_int[-which(num_int%in%unused_var)]

# random forest
library("caret")
library("ggplot2")
training_mod<-training[,var]
training_mod<-data.frame(training_mod, classe=training$classe)
testing_mod<-testing[,var]
testing_mod<-data.frame(testing_mod, classe=testing$classe)
validation_mod<-validation[,var]
validation_mod<-data.frame(validation_mod, classe=validation$classe)



# random forest
Sys.time()
modfit_rf_all<-train(classe~., data=training_mod, method="rf", prox=TRUE, preProcess="pca", na.remove=TRUE, trControl=trainControl(preProcOptions = list(thresh = 0.8)))
pred_rf_all<-predict(modfit_rf_all, testing_mod)
table(pred_rf_all, testing_mod$classe)
confusionMatrix(testing_mod$classe, pred_rf_all)[[3]][1] # accuracy 0.92
Sys.time()

# lda - 0.42
# nb - 0.52
# glm - 0.85 - то же, что и опорные вектора
# gbm - 0.73 verbose=FALSE

modfit_lda_all<-train(classe~., data=training_mod, method="lda", preProcess="pca", na.remove=TRUE, trControl=trainControl(preProcOptions = list(thresh = 0.8)))
pred_lda_all<-predict(modfit_lda_all, testing_mod)
confusionMatrix(testing_mod$classe, pred_lda_all)[[3]][1] # accuracy 0.85

# ансамбль моделей
# в ансамбле - 4 модели
predDF<-data.frame(pred_rf_all, pred_svm_all, pred_gbm_all, pred_nb_all, classe=testing_mod$classe)
combModFit<-train(classe~., method="rf", data=predDF)
combPred<-predict(combModFit, predDF)
# gam - 0.45
# rf - 0.92
# nb - 0.88
# lda - 0.92
# gbm - 0.92

# в ансамбле - 2 модели
predDF<-data.frame(pred_rf_all, pred_svm_all, classe=testing_mod$classe)
combModFit<-train(classe~., method="gam", data=predDF)
combPred<-predict(combModFit, predDF)
confusionMatrix(testing_mod$classe, combPred)[[3]][1]
# gam - 0.45
# rf - 0.92
# nb - 0.91
# lda - 0.92
# gbm - 0.92
