library('pROC')
#install.packages("randomForest")
library('randomForest')

titanic= read.csv('H:/My Drive/20230401_KAGGLE/titanic/train.csv')
head(titanic)


dim(titanic)
summary(titanic)
str(titanic)

 # apply(
 #   titanic[, match(c('Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'),
 #       colnames(titanic))],
 #   2,function(x){sum(is.na(x))})

#filter out any rows with NA in relevant columns
 # anyNA_ind= apply(
 #   titanic[,match( c('Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'),
 #                 colnames(titanic))],
 #   1,function(x){any(is.na(x))})
# tit_filt= titanic[!anyNA_ind,]
# dim(tit_filt)

#NAs are only in 'Age' -> impute median
tit_filt= titanic
tit_filt$Age[is.na(tit_filt$Age)]= median(tit_filt$Age,na.rm=TRUE)

#split into DF_train and DF_eval
set.seed(123)
accur_vec= c()
best_mod_acc=0

for(i in 1:10) {
  train_idx= sample(1:dim(tit_filt)[1],size=ceiling(dim(tit_filt)[1]*.8),replace=FALSE)  #80/20 split
  eval_idx= c(1:dim(tit_filt)[1])[-train_idx]
  tit_train= tit_filt[train_idx,]
  tit_eval= tit_filt[eval_idx,]
  
  #full model w/o interactions
  # tit_mod= glm(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked, data=tit_filt,family='binomial')
  # summary(tit_mod)
  
  #automated feature selection
  # null_mod= glm(Survived~1,data=tit_train,family='binomial')
  # tit_mod= step(null_mod, direction='forward', scope=list(lower=null_mod,upper=tit_mod))
  # summary(tit_mod)
  
  #randomForest
  tit_mod= randomForest(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked,data=tit_train,ntree=100,
               mtry=ceiling(dim(tit_train)/5))
  
  #evaluate how model is doing
  tit_prob= predict(tit_mod,newdata=tit_eval, type='response')
  surv_pred= (tit_prob>0.5)
  (accur= sum(surv_pred==tit_eval$Survived)/length(tit_eval$Survived))
  accur_vec[i]=accur
  
  #store the best model
  if(accur > best_mod_acc){
    best_mod_acc= accur
    best_mod= tit_mod
  }
  
  tit_ROC= roc(predictor= tit_prob, response= tit_eval$Survived)
  plot(tit_ROC)
}

boxplot(accur_vec,ylim=c(0,1),ylab='accuracy')

#predictions for test set
test= read.csv('H:/My Drive/20230401_KAGGLE/titanic/test.csv')
head(test)
apply(
  test[,match( c('Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'), colnames(test))],
  2,function(x){sum(is.na(x))})
test$Age[is.na(test$Age)]= median(test$Age,na.rm=TRUE)
test$Fare[is.na(test$Fare)]= median(test$Fare,na.rm=TRUE)

test_prob= predict(newdata=test,best_mod,type='class')
plot(sort(test_prob))
test_pred= test_prob>0.5
summary(test_pred)

out= cbind.data.frame(PassengerId=test$PassengerId,Survived=as.numeric(test_pred))
write.csv(out,'H:/My Drive/20230401_KAGGLE/titanic/20230421_randForestPred.csv',row.names=FALSE)
