d <- read.csv("~/Downloads/Output_Final_Chicago.csv")
str(d)

library(gmodels)
library(caret)
library(uplift)
library(dplyr)
options(java.parameters = "-Xmx64048m") # 64048 is 64 GB


#removing non-related or perfectly colinear features
d$F1<-NULL
d$Vendor<-NULL
d$Units<-NULL
d$Market_Name<-NULL

table(d$PR)
table(d$Week)
table(d$SY)
table(d$GE)
table(d$Item)

#removing items that have less that 1000 observations (0.3% of data)
d<-d %>% group_by(Item) %>% filter(n()>= 1000)
d<-as.data.frame(d)

table(d$Outlet)
table(d$Est_Acv)
table(d$Open)
table(d$Closed)
table(d$Masked_Name)

d$Week <- as.factor(d$Week)
d$SY <- as.factor(d$SY)
d$GE <- as.factor(d$GE)
d$F <- as.factor(d$F)
d$D <- as.factor(d$D)
d$Outlet <- as.factor(d$Outlet)
d$Open <- as.factor(d$Open)
d$Closed <- as.factor(d$Closed)
d$Masked_Name <- as.factor(d$Masked_Name)

#Frequency Encoding IRI Keys
IRI<-as.data.frame(table(d$IRI_KEY))
names(IRI)=c("IRI_KEY","IRI_Freq")
d<-merge(x = d, y = IRI, by = "IRI_KEY", all.x = TRUE)
d$IRI_KEY<-NULL

#min-max normalization IRI_Freq
d$IRI_Freq<-(d$IRI_Freq-min(d$IRI_Freq))/(max(d$IRI_Freq)-min(d$IRI_Freq))

#Frequency Encoding Items
Item<-as.data.frame(table(d$Item))
names(Item)=c("Item","Item_Freq")
d<-merge(x = d, y = Item, by = "Item", all.x = TRUE)
d$Item<-NULL

#min-max normalization Items and Acv
d$Item_Freq<-(d$Item_Freq-min(d$Item_Freq))/(max(d$Item_Freq)-min(d$Item_Freq))
d$Est_Acv<-(d$Est_Acv-min(d$Est_Acv))/(max(d$Est_Acv)-min(d$Est_Acv))

str(d)

#creating a variable to save stats of dollars
dstat<-summary(d$Dollars)
dstat

#y (column) variable creation
y<- as.numeric(d$Dollars>=dstat["Median"])
d<-cbind(y,d)
d$Dollars<-NULL
d$y_num<-y
d$y<-ifelse(d$y==1,"Y","N")
d$y<-as.factor(d$y)
d$y <- relevel(d$y,"Y")
rm(y)

# summary statistics:
CrossTable(d$PR, d$y)

dummy <- dummyVars(~ Week, data = d)
d<-cbind(d,predict(dummy, d))
d$Week<-NULL

dummy <- dummyVars(~ SY, data = d)
d <- cbind(d,predict(dummy, d))
d$SY<-NULL

dummy <- dummyVars(~ GE, data = d)
d <- cbind(d,predict(dummy, d))
d$GE<-NULL

dummy <- dummyVars(~ F, data = d)
d <- cbind(d,predict(dummy, d))
d$F<-NULL

dummy <- dummyVars(~ D, data = d)
d <- cbind(d,predict(dummy, d))
d$D<-NULL

dummy <- dummyVars(~ Outlet, data = d)
d <- cbind(d,predict(dummy, d))
d$Outlet<-NULL

dummy <- dummyVars(~ Open, data = d)
d <- cbind(d,predict(dummy, d))
d$Open<-NULL

dummy <- dummyVars(~ Closed, data = d)
d <- cbind(d,predict(dummy, d))
d$Closed<-NULL

dummy <- dummyVars(~ Masked_Name, data = d)
d <- cbind(d,predict(dummy, d))
d$Masked_Name<-NULL

rm(dummy)

# Find if any linear combinations exist and which column combos they are.
# Below I add a vector of 1s at the beginning of the dataset. This helps ensure
# the same features are identified and removed.

# first save response
y <- d$y

# create a column of 1s. This will help identify all the right linear combos
d <- cbind(rep(1, nrow(d)), d[2:ncol(d)])
names(d)[1] <- "ones"

# identify the columns that are linear combos
comboInfo <- findLinearCombos(d)
comboInfo

# remove columns identified that led to linear combos
d <- d[, -comboInfo$remove]

# remove the "ones" column in the first column
d <- d[, c(2:ncol(d))]

# Add the target variable back to our data.frame
d <- cbind(y, d)

rm(y, comboInfo)  # clean up

#removing near zero variance columns
nzv <- nearZeroVar(d, saveMetrics = TRUE)
d <- d[, c(TRUE,!nzv$zeroVar[2:ncol(d)])] #No feautres had zero variance
#d <- d[, c(TRUE,!nzv$nzv[2:ncol(d)])] #Over 50% had nzv = True
rm(nzv)

str(d)
d<-d[-209767,]

set.seed(1234)
d_cdp <- createDataPartition(d$y, p = 0.6,list = FALSE)
d_train<- d[d_cdp,]
d_test<- d[-d_cdp,]
rm(d_cdp)

Yhat_1 <- train(y ~ . -y_num + trt(PR)
                , method = "glm"
                , data = d_train
                , family = "binomial")

# create a test set with the treatment field reversed
d_test2 <- d_test
d_test2$PR<-1-d_test2$PR

#to check if the are indeed different
sum(d_test[,"PR"]!= d_test2[,"PR"])


pred1 <-predict(Yhat_1, newdata = d_test, type='prob')
pred2 <-predict(Yhat_1, newdata = d_test2, type='prob')

uplift = data.frame(pred1$Y, pred2$Y
                    , Uplift = pred1$Y-pred2$Y)


# Transactions that have the greatest positive change due to PR
uplift <- uplift[order(-uplift$Uplift),]
head(uplift)

# If you didn't want to "waste" resourses on transactions that already likely to
#be above median then you could igore them, and focus on transactions that have the
#greatest positive change
uplift2 <- subset(uplift, uplift$pred1.Y<=0.5)
uplift2 <- uplift2[order(-uplift2$Uplift),]
head(uplift2)


##################### h2o ###########################################################
# load package
library(h2o)

# start a one-node h2o cluster on your local machine.
# By default, your h2o instance will be allowed to use all your cores and
# 25% of your system memory unless you specify otherwise.
h2o.init()
h2o.clusterInfo()
# If you wanted to use an H20 (in a multi-node Hadoop environment), you need to
# specify the IP and Port for that established cluster.
# h2o.init(ip = "123.45.67.89", port = 54321)

# If you want to specify cores and memory to use you can like so:
#h2o.shutdown()                          # shutdown your cluster if running
#Y
#h2o.init(nthreads=2, max_mem_size="4g") # specify what you want
#h2o.clusterInfo()                       # inspect cluster info


train <- as.h2o(d_train[,-6])
test <- as.h2o(d_test[,-6])
test2 <- as.h2o(d_test2[,-6])

y <- "y"                                # target variable to learn
x <- setdiff(names(train), y)                # features are all other columns


library(caret)
auto <- h2o.automl(x=x, y=y,
                   training_frame = train,
                   max_models = 100,
                   seed=1)

h2o.performance(auto@leader,test)

pred3<- h2o.predict(auto,test)
pred4<- h2o.predict(auto,test2)

pred3<-as.data.frame(pred3)
pred4<-as.data.frame(pred4)

uplift2 = data.frame(pred3, pred4
                     , Uplift = pred3$Y-pred4$Y)

table(uplift2$predict.2)

# shutdown your cluster
h2o.shutdown()



######

# logit model preds
trP_logit <- predict(Yhat_1 , newdata=d_train, type='prob')[,1]
trC_logit <- predict(Yhat_1 , newdata=d_train)
teP_logit <- predict(Yhat_1 , newdata=d_test, type='prob')[,1]
teC_logit <- predict(Yhat_1 , newdata=d_test)
# lda model preds
trP_rf <- as.data.frame(h2o.predict(auto,train))[,3]
trC_rf <- as.data.frame(h2o.predict(auto,train))[,1]
teP_rf <- as.data.frame(h2o.predict(auto,test))[,3]
teC_rf <- as.data.frame(h2o.predict(auto,test))[,1]


library(pROC)
plot(roc(d_test$y, teP_logit), col = "red")
plot(roc(d_test$y, teP_rf), col = "blue",add=TRUE,lwd=1)
legend(0, 1, legend=c("Logit","RF"), col=c("red","blue"), lty=1:5,cex=1)

AUCtable = rbind(auc(roc(d_test$y, teP_logit)),
                 auc(roc(d_test$y, teP_rf)))
rownames(AUCtable) <- c("Logit_test","RF_test")
colnames(AUCtable) = "AUC"
AUCtable



################### Using the uplift package ##################################

d_train3<-d_train[,-1]
d_test3<-d_test[,-1]

str(d_train3)

# use upliftRF to apply a Random Forest (alternatively use upliftKNN() to apply kNN).
up.fit <- upliftRF(y_num ~ . + trt(PR)
                   , data = d_train3[1:500,]
                   , mtry = 3             # number of variables to be tested in each node
                   , ntree = 50          # number of trees to generate in the RF
                   , split_method = "KL"  # split criteria used at each node of each tree
                   , minsplit = 200       # minimum number of observations that must exist in a node in order for a split to be attempted
                   , verbose = TRUE       # print status messages?
)

# predict.upliftRF(up.fit, newdata = d_test3)

# d_train3 <- sim_pte(n = 2000, p = 20, rho = 0, sigma =  sqrt(2), beta.den = 4)
# d_train3 <- ifelse(d_train3$PR == 1, 1, 0)  
# pred <- predict(up.fit, d_train3)    
# head(pred)

# pred below was throwing errors, so i tried leveling, but we got no factor variable lol
#for(i in ncol(d_train3))
#{
#  levels(d_test3[,i]) <- levels(d_train3[,i])
#}

# try 2
#d_test3 <- rbind(d_train3[1, ] ,d_test3)
#d_test3 <- d_test3[-1,]

pred <- data.frame(predict(up.fit, newdata = d_test3))

#CHECK SENSITIVITY AND SPECIFICITY FOR CANDIDATE MODEL
table(d_test$y,pred3$predict)