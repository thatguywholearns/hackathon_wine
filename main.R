#-------------------------------------Set-up environment----------------------------------------#
# Import packages
pacman::p_load(pacman, tidyverse, caret, rstudioapi, corrplot, yardstick, reshape2, RColorBrewer)

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

#-----------------------------------------functions------------------------------------------#



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

# Create matrix with only features and check correlations
feat <- as.data.frame(df[, -ncol(df)])
cor(feat)

# Visualize correlations independant variables 
corrplot(cor(feat))

# Visualize dependant variable
ggplot(df, aes(x = quality)) +
  geom_bar()

#-----------------------------Preprocessing and feature selection------------------------------#



#---------------------------------------------Modelling--------------------------------------#

# Split in train and test data sets
indexes <- createDataPartition(df$quality,
                               times = 1,
                               p = 0.70,
                               list = F)
df_train <- df[indexes, ]
df_val <- df[-indexes, ]

models = c("knn", "rf", "gbm", "svmPoly", "svmRadial")
models_fitted <-  list()
models_results <- list()
aggr_confusion_matrix <- list()

# Set-up resampling method
fit_control <-  trainControl(method = "repeatedcv",
                             number = 5,
                             repeats = 1,
                             sampling = "up",
                             returnResamp = "all",
                             savePredictions = T,
                             returnData = T)

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

# Define theme for lay out
theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)

# Boxplot of results
bwplot(resamps, layout = c(2, 1))

#-----------------------------------Error Analysis----------------------------------#

# Look at confusionmatrix for a specific model
confustion_mat <- models_fitted$rf$pred %>% 
  filter(Resample == "Fold1.Rep1") %>%
  conf_mat(obs, pred)

# Plot confusion matrix for this 
autoplot(confustion_mat, type = "heatmap")

# Average confusion matrix across all folds in terms of the proportion of the data contained in each cell.
cells_per_resample  <- models_fitted$rf$pred %>% 
  group_by(Resample) %>%
  conf_mat(obs, pred) %>%
  mutate(tidied = map(conf_mat, tidy)) %>%
  unnest(tidied)

counts_per_resample <- models_fitted$rf$pred %>%
  group_by(Resample) %>%
  summarise(total = n()) %>%
  left_join(cells_per_resample, by = "Resample") %>%
  mutate(prop = value/total) %>%
  group_by(name) %>%
  summarize(prop = mean(prop))

# Express in percentage
counts_per_resample_perc <- counts_per_resample
counts_per_resample_perc$prop <- round(counts_per_resample_perc$prop * 100, digits = 2)

# Convert to matrix
mean_confusion_mat <- matrix(counts_per_resample_perc$prop, byrow = TRUE, ncol = 7)

# Convert to data frame
mean_confusion_df <- as.data.frame(mean_confusion_mat)
mean_confusion_df$id <- c("3", "4", "5", "6", "7", "8","9")
colnames(mean_confusion_df) <- c("3", "4", "5", "6", "7", "8","9","pred")

# Melt the data frame
mean_confusion_melt <- melt(mean_confusion_df, variable.name = "true")

# Plot as heatmap
ggplot(mean_confusion_melt, aes(x=true, y=pred, fill=value)) +
  geom_tile() +
  scale_fill_distiller(palette="Greens", direction=1) +
  geom_text(aes(label=value), color="black") +
  labs(title = "Confusion matrix averaged across cross validation")



# importance <- varImp(SVModel, scale=FALSE)
# plot(importance)


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

# gbmFit4 <- train(Class ~ ., data = training, 
#                  method = "gbm", 
#                  trControl = fitControl, 
#                  verbose = FALSE, 
#                  ## Only a single model can be passed to the
#                  ## function when no resampling is used:
#                  tuneGrid = data.frame(interaction.depth = 4,
#                                        n.trees = 100,
#                                        shrinkage = .1,
#                                        n.minobsinnode = 20),
#                  metric = "ROC")
# 
# importance <- varImp(SVModel, scale=FALSE)
# plot(importance)
