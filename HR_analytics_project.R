library(ISLR)
library(pROC)
library(caret)
library(tree)

hrdata = read.csv('B:\\MS BAIM\\Fall\\Predictive Analytics\\Project\\HR_analytics_data_1000.csv'
                  ,header = T
                  ,sep = ","
                  ,colClasses = c("factor","factor","factor","factor","factor","factor","factor"
                                  ,"character","character"
                                  ,"factor","factor","factor","factor","factor","numeric"
                                  ,"character"
                                  ,"factor","factor","factor","factor"
                                  ,"character"
                                  ,"factor","factor","factor","factor"
                                  ,"numeric","numeric","numeric","numeric","numeric"
                                  ,"numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric"
                                  ,"factor","factor"
                                  ,"factor","factor","numeric","numeric"
                                  ,"numeric","numeric")
                  )

str(hrdata)
names(hrdata)

hrdata[hrdata$performance2014level<0,]$performance2014level=0;
hrdata[hrdata$performance2013level<=0,]$performance2013level=0.1;
hrdata$performancechange = hrdata$performance2014level/hrdata$performance2013level;

cor(hrdata[,c('Hourly.Rate','X2013.SPH.Actual','X2014.SPH.Actual'
              ,'X2013.SPH.Goal','X2014.SPH.Goal','Pnormal'
              ,"Pflag_below","Pflag_above"
              # ,'P0','P25','P50','P75','P100'
              )])

par(mfcol = c(1,1))
pairs(hrdata[,c('Hourly.Rate','Pnormal'
                ,'performance2013level','performance2014level'
                # ,'X2013.SPH.Actual','X2014.SPH.Actual'
                # ,'X2013.SPH.Goal','X2014.SPH.Goal'
              # ,"Pflag_below","Pflag_above"
              # ,'P0','P25','P50','P75','P100'
              )]
      ,cex.labels = 1.5)
 
# hrdata$perfomance2013level = hrdata$X2013.SPH.Actual/hrdata$X2013.SPH.Goal
# hrdata$perfomance2014level = hrdata$X2014.SPH.Actual/hrdata$X2014.SPH.Goal

summary(hrdata)

############# dropping some columns #############
drops = c("Hire.Date","Last.Hire.Date","Effective.Date","Vol.Invol","SP"
          ,'Open.Date','P0','P25','P50','P75','P100','EMPLID','Location'
          ,'Period','Ethnic','Type','Gender','Grade','FT.PT','Action.Reason'
          ,'Ethnic.Desc','Market.Type','MT.Desc','Store.Class','left_numeric'
          ,'still_employed','still_employed_numeric')
hrdatasubset = hrdata[,!names(hrdata) %in% drops]

########### Training and Test Dataset ##########
set.seed(2016)

levels(hrdata$left) <- make.names(levels(factor(hrdata$left)))
# hrdata$left <- relevel(hrdata$left,1)

inTrain=createDataPartition(hrdata$left,p=0.70,list=F)
train=hrdata[inTrain,]
test=hrdata[-inTrain,]
dim(train)
names(train)
summary(train)

str(train)

############# DECISION TREE ################

############# Applying Decision Tree ##############
set.seed(2016)
#install.packages("tree")
library(tree)
library(randomForest)
library(e1071)

treefit_1 = tree(left ~ Job.Title+Region+Pflag_below+Pflag_above+Pnormal+Hourly.Rate
               +performance2013level+performance2014level+performancechange
               +TMperformance2013+TMperformance2014
               ,control = tree.control(nobs = nrow(train)
                                       ,mincut = 0
                                       ,minsize = 1
                                       ,mindev = 0.01)
               ,data = train)
summary(treefit_1)

plot(treefit_1);
text(treefit_1, pretty = 0)
boxplot(performance2014level~TMperformance2014,data = hrdata)
boxplot(performancechange~left,data = hrdata)


library(caret)
levels(train$left) = make.names(levels(factor(train$left)))

ctrl = trainControl(method = 'cv'
                    #,repeats = 5
                    #,number = 20
                    ,classProbs = TRUE
                    ,summaryFunction = twoClassSummary)

treefit_2=train(left ~ Job.Title+Pflag_below+Pflag_above+Pnormal+Hourly.Rate
                +performance2013level+performance2014level
                +TMperformance2013+TMperformance2014
                ,data = train
                ,method = 'glm'
                ,family='binomial'
                ,trControl = ctrl
                #,preProcess=c("center","scale")
                #,nIter=10
                ,metric = 'ROC')

treefit_2 = tree(left ~ Job.Title+Pflag_below+Pflag_above+Pnormal+Hourly.Rate
                 +performance2013level+performance2014level
                 +performancechange
                 ,control = tree.control(nobs = nrow(train)
                                         ,mincut = 0
                                         ,minsize = 1
                                         ,mindev = 0.01)
                 ,data = train)
summary(treefit_2)
plot(treefit_2);
text(treefit_2, pretty = 0);

plot(treefit_2$finalModel);
text(treefit_2$finalModel, pretty = 0);

################ Pruning the tree ##################
head((train$left))
cv.hrdatasubset = cv.tree(treefit_2)
cv.hrdatasubset

plot(x=cv.hrdatasubset$size,y=cv.hrdatasubset$dev,type = 'b', col = "blue"
     ,xlab = "Number of Terminal Nodes",ylab = "Cross-validated error")

prunedfit = prune.tree(treefit_2, best = 3)
summary(prunedfit)

plot(prunedfit);text(prunedfit, pretty = 0)


############ Random forest and SVM #################

rf = randomForest(left ~ Job.Title+Region+Pflag_below+Pflag_above+Pnormal+Hourly.Rate
                  +performance2013level+performance2014level
                  +TMperformance2013+TMperformance2014
                  # ,mtry = 5
                  # ,ntree = 200
                  ,replace = TRUE
                  # ,sampsize = 300
                  ,data = train)
summary(rf)
plot(rf,type = "simple")
svmfit = svm(left ~ Pnormal+Hourly.Rate
             +performance2013level+performance2014level
             ,data = train
             ,kernel = "radial"
             ,cost=100     #best performing value as checked using tune() function
             ,gamma=0.05   #best performing value as checked using tune() function
             ,scale = FALSE
             ,probability = TRUE)

summary(svmfit)
############### svm using caret package #################

library(caret)
levels(train$left) = make.names(levels(factor(train$left)))

ctrl = trainControl(method = 'repeatedcv'
                    ,repeats = 5
                    ,number = 20
                    ,classProbs = TRUE
                    ,summaryFunction = twoClassSummary)
svmfit=train(left ~ Job.Title+
              Pnormal+Hourly.Rate
             +performance2013level+performance2014level
             +TMperformance2013+TMperformance2014
             ,data = train
             ,method = 'svmRadial'
             ,trControl = ctrl
             ,preProcess=c("center","scale")
             ,sigma = 0.01
             ,cost = list(c(0.5,1,1.5))
             ,metric = 'ROC')

tune.out = tune(svm
                ,left~ Pnormal+Hourly.Rate
                +performance2013level+performance2014level
                ,data=train
                ,kernel = "radial"
                ,ranges=list(cost = c(0.1,1,10,100,10000)
                            ,gamma = c(0.005,0.05,0.5,1,2))
)
summary(tune.out)
tune.out$best.model



############### Validating models' Performance ###########
############### svm #######################
testYhatsvm = predict(svmfit,type = "prob",newdata = test)
testYhatsvm[testYhatsvm[,2]>0.5,2] = 1
testYhatsvm[testYhatsvm[,2]<=0.5,2] = 0
confusionMatrix(data = testYhatsvm, test$left, positive = '1')


######## tree and random forest ###########
testYhatTree = predict(treefit_2,type = 'prob',newdata = test)
testYhatTree[testYhatTree[,2]>0.5,2] = 1
testYhatTree[testYhatTree[,2]<=0.5,2] = 0

testYhatforest = predict(rf,type = "prob",newdata = test)
testYhatforest[testYhatforest[,2]>0.5,2] = 1
testYhatforest[testYhatforest[,2]<=0.5,2] = 0

#testYhatsvm = predict(svmfit,type = "prob",newdata = test)
testYhatsvm = predict(svmfit,type = "prob",newdata = test)
testYhatsvm[testYhatsvm[,1]>0.5,1] = 1
testYhatsvm[testYhatsvm[,1]<=0.5,1] = 0

testYhatTree
testYhatforest
testYhatsvm

confusionMatrix(data = testYhatTree[,2], test$left_numeric, positive = '1')
confusionMatrix(data = testYhatforest[,2], test$left_numeric, positive = '1')
confusionMatrix(data = testYhatsvm, test$left_numeric, positive = '1')

plot(testYhatTree[,2],test$left,col='blue'
     ,main = "Actual Turnover vs Predicted Turnover"
     ,xlab = "Predicted Turnover",ylab = 'Actual Turnover')

TestErrorTree = postResample(pred = testYhatTree, obs = test$left_numeric)
TestErrorTree[[1]]^2

treefit_1_probs <- predict(treefit_1,newdata=test)[,2]
treefit_2_probs <- predict(treefit_2,type = 'prob',newdata=test)[,2]
prunedfit_probs <- predict(prunedfit,newdata=test)[,2]
forestfit_probs <- predict(rf,type = "prob",newdata = test)[,2]
svm_probs <- predict(svmfit,type = "prob", newdata = test)
###################### ROC curve #########################
library(pROC)
rocCurve_1=roc(response=test$left
               ,predictor = treefit_1_probs
               ,levels=rev(levels(test$left)))

rocCurve_2=roc(response=test$left
               ,predictor = treefit_2_probs
               ,levels=rev(levels(test$left)))

rocCurve_3=roc(response=test$left
               ,predictor = prunedfit_probs
               ,levels=rev(levels(test$left)))

rocCurve_4=roc(response=test$left
               ,predictor = forestfit_probs
               ,levels=rev(levels(test$left)))

rocCurve_5=roc(response=test$left
               ,predictor = svm_probs[,2]
               ,levels=rev(levels(test$left)))
par(mfrow=c(1,1)) #reset plot graphics to one plot

plot(rocCurve_2
     ,legal.axes=T
     ,col="red"
     ,main="Receiver Operating Characterstic (ROC) Curve"
     )

lines(rocCurve_2,col = 'blue')
lines(rocCurve_3,col = 'green')
lines(rocCurve_4,col = 'violet')

#?legend
legend("bottomright"
       ,inset=0,title="Model"
       ,border="black"
       ,bty="n"
       ,cex=0.8
       ,legend=c("Model 1","Model 2","Model 3","Random Forest")
       ,fill=c("red","blue","green","violet"))

###################### Step-9 #########################
auc(rocCurve_1)
auc(rocCurve_2)
auc(rocCurve_3)
auc(rocCurve_4)
auc(rocCurve_5)

################ Descriptive Analysis #################
######## Terminations by year and region #########
length(unique(hrdata$Region))
df = hrdata
df$left = as.numeric(df$left)
df1 = aggregate(left ~ Year + Region
                ,data = df
                ,FUN = 'sum')
head(df1)
df1$Year = as.factor(df1$Year)
plot(x=df1$Year, y=df1$left, data = df1
     ,xlab = "Year"
     ,ylab = "Terminations in a Region"
     ,main = "Terminations by Year")

p = ggplot(df1,aes(x = factor(df1$Year),y = df1$left))
p = p + geom_boxplot(fill = factor(df1$Year))
p
############# promoted or demoted ################
df = aggregate(cbind(left_numeric,still_employed_numeric) ~ TMperformance2013 + TMperformance2014
               ,data = hrdata
               ,FUN = "sum")
colnames(df) = c('TMperformance2013','TMperformance2014','left','Still Employed')

write.table(df,'B:\\MS BAIM\\Predictive Analytics\\Project\\PerformanceUpgradeData.csv'
            ,sep = ',',row.names = FALSE)

############ job title based on total data ###############
df1 = aggregate(cbind(left_numeric,Pflag_above,Pflag_below) ~ Job.Title
               ,data = hrdata
               ,FUN = "sum")
df2 = aggregate(cbind(Hourly.Rate,Pnormal) ~ Job.Title
               ,data = hrdata
               ,FUN = "mean")
df3 = aggregate(cbind(left) ~ Job.Title
                ,data = hrdata
                ,FUN = "length")
colnames(df3) = c("Job.Title","Total Employees")

df4 = merge(df1,df2, by = 'Job.Title',all.x = TRUE,all.y = TRUE)
df = merge(df4,df3,by = 'Job.Title',all.x = TRUE,all.y = TRUE)

write.table(df,'B:\\MS BAIM\\Predictive Analytics\\Project\\Jobtitle_total_data.csv'
            ,row.names = FALSE,sep = ',')

############ job title based on left employees data ###############
hrdataleft = hrdata[hrdata$left_numeric==1,]
df1 = aggregate(cbind(left_numeric,Pflag_above,Pflag_below) ~ Job.Title
                ,data = hrdataleft
                ,FUN = "sum")
df2 = aggregate(cbind(Hourly.Rate,Pnormal) ~ Job.Title
                ,data = hrdataleft
                ,FUN = "mean")
df3 = aggregate(cbind(left) ~ Job.Title
                ,data = hrdataleft
                ,FUN = "length")
colnames(df3) = c("Job.Title","Total Employees")

df4 = merge(df1,df2, by = 'Job.Title',all.x = TRUE,all.y = TRUE)
df = merge(df4,df3,by = 'Job.Title',all.x = TRUE,all.y = TRUE)

write.table(df,'B:\\MS BAIM\\Predictive Analytics\\Project\\Jobtitle_left_data.csv'
            ,row.names = FALSE,sep = ',')

