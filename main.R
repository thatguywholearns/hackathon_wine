#-------------------------------------Set-up environment----------------------------------------#
# Import packages
pacman::p_load(pacman, tidyverse, caret, rstudioapi, DMwR, ROSE, corrplot)

# Set working directory
path <- getActiveDocumentContext()$path
setwd(dirname(path))

# Set random seed
set.seed(123)

# Set up parallel processing

# Find how many cores are on your machine
detectCores()

# Create Cluster with desired number of cores. Don't use them all! Your computer is running other processes. 
clusters <-  makeCluster(2)

# Register cluster
registerDoParallel(clusters)

# Confirm how many cores are now "assigned" to R and RStudio
getDoParWorkers() # Result 2 

#------------------------------------------Read data-------------------------------------------#

# Load the data
df <- read.csv(file = "./data/training.csv")
testing <- read.csv(file = "data/validation.csv")

#-------------------------------------Initial Exploration--------------------------------------#

# Drop index column ?? do with read function if possible
df <- df[, -1]
testing <- testing[, -1]

# Check structure of the data
str(df)
str(testing)

# Rename columns
new_names <- c("fixed_acidity",  
               "volatile_acidity", 
               "citric_acid", 
               "residual_sugar", 
               "chlorides", 
               "free_sulfur_dioxide", 
               "total_sulfur_dioxide", 
               "density", 
               "pH",  
               "sulphates", 
               "alcohol", 
               "quality") 

colnames(df) <- new_names
colnames(testing) <- new_names[-length(new_names)]

# Check for NA's
sum(is.na(df))
sum(is.null(df))

# Check for duplicates
df[-duplicated(df), ]

# Convert class to factor
df$quality <- as.factor(df$quality)

# Visualize distributions of the independant variables
df[, -ncol(df)] %>% 
  gather() %>% 
  ggplot(aes(x = value)) +
    facet_wrap(~ key, scales = "free") +
    geom_density()

# Visualize distributions of the independant variables on by one
plt <- list()

for (i in 1:(ncol(df) - 1)) {
  names <- colnames(df)[i]
  plt[[names]] <- ggplot(data =  df, aes(df[, i])) +
    geom_density() +
    labs(title = paste("Density plit", colnames(df)[i]),
         x = colnames(df)[i])
  print(plt[names])
}

# Visualize distributions of the independant variables with class
df %>% 
  gather(key = "variable", value = "measurement", -quality) %>% 
  ggplot(aes(x = measurement, fill = quality)) +
    facet_wrap(~ variable, scales = "free") +
    geom_histogram()

# Visualize correlations independant variables 
corrplot(df, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45)

#--------------------------------Preprocessing and feature selection--------------------------------#



#---------------------------------------------Modelling--------------------------------------#

# Split in train and test data sets
indexes <- createDataPartition(df$quality,
                               times = 1,
                               p = 0.70,
                               list = F)
df_train <- df
df_val <- df[-indexes, ]

test <- data.frame(model.matrix(quality~., data = df_train))[,-1]
View(test)

# # Balance the train data
# df_train <-  downSample(x = df_train[, -ncol(df_train)],
#                         y = df_train$quality,
#                         yname = quality)
# 
# df_train <- upSample(x = df_train[, -ncol(df_train)],
#                      y = df_train$quality,
#                      yname = quality)
# 
# df_train <- ROSE(quality ~ ., data  = df_train)$data

# Create models and predict on train
# models = c("rf", "knn", "svmRadial")
models = c("knn", "rf", "svmRadial")
models_fitted <-  list()
models_results <- list()
aggr_confusion_matrix <- list()

# Set-up resampling method
fit_control <-  trainControl(method = "repeatedcv",
                             number = 5,
                             repeats = 1,
                             sampling = "up")

#"none", "down", "up", "smote", or "rose"


for (i in models) {
  
  model <- train(quality ~ ., 
                 data = df_train,
                 method = i,
                 preProc = c("center", "scale"),
                 trControl = fit_control,
                 metric = "Accuracy")
  
  models_fitted[i] <- list(model)
  
  models_results[i] <- list(model$results)
  
}

# Check model performance and pick best performing model
resamps <- resamples(models_fitted)
theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
bwplot(resamps, layout = c(2, 1))

#---------------------------------------Error Analysis------------------------------------#

#--------------------------------------Predict--------------------------------------#

# random forest has te best performance. We train a random forest on all the data
model <- train(quality ~ ., 
               data = df,
               method = "rf",
               preProc = c("center", "scale"),
               trControl = fit_control,
               metric = "Accuracy")

# Use trained model to make predictions
final_pred <-  predict(model, testing)

# Save results to csv
write.csv(final_pred, "./final_pred.csv")

# Reset settings for parallel computing
stopCluster(clusters)
