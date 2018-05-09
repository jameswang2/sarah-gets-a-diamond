
# WARNING: This script assumes that you have some basic knowledge about how to 
# run R code. If you do not have a very good understanding, you can quickly learn 
# by reviewing R script used in the Intro to R Programming. The link to that file 
# is: https://github.com/DardenDSC/intro-to-r-programming/blob/master/script.R


# install R packages if you don't already have them ----------------------------
install.packages("tidyverse")
install.packages("rpart")
install.packages("rpart.plot")


# setup your R environment -----------------------------------------------------
options(scipen=999, digits=6)
suppressMessages(library(tidyverse))
suppressMessages(library(rpart))
suppressMessages(library(rpart.plot))

# read the data ----------------------------------------------------------------
raw_diamond_dat <- read_csv("https://raw.githubusercontent.com/DardenDSC/sarah-gets-a-diamond/master/data/diamond-data.csv", 
                            col_types = cols())

# create a new variable to keep the raw data separate
# this way we always have an original copy of the data to fall back on in case 
# we make any mistakes while messing with a formatted copy
diamond_dat <- raw_diamond_dat


# prepare the data for modeling ------------------------------------------------

# If you remember back in Decision Analysis (DA) class you must first explore the 
# data, format it, determine if variables should be transformed to make it more linear. 
# This step is typically called "data munging" or "data wrangling" and is said to 
# constitute ~80% of the time dedicated towards building a predictive model. Once you 
# have the data in a good format, it is easy to train a model. Let's start by doing that 
# cleaning process.

# In this step the "mutate()" function creates new variables based on transformations 
# of existing variables. log(Price) is creating the log transform of the price variable.

diamond_dat <- diamond_dat %>%
  mutate(log_price = log(price),
         log_carat_weight = log(carat_weight),
         carat_weight_binned = cut(carat_weight, 
                                   breaks = c(0,1.5,2,Inf), 
                                   labels = c("0.00-1.49", "1.50-1.99", "2.00+"),
                                   right = FALSE))

# If we are trying to predict price, then it makes sense to first compare that with 
# carat weight. A diamond's value is related to how big it is. An initial scatterplot 
# of the Price vs. Carat Weight shows that Carat Weight typically falls into three 
# distinct buckets: 0-1.5, 1.5-2, and 2+. Also, there is heteroskedasticity (non-constant variance) 
# that is happening with each bucket. The variance is getting amplified.

ggplot(data=diamond_dat, mapping=aes(x=carat_weight, y=price)) +
  geom_point(aes(color=carat_weight_binned)) + 
  labs(x = "Carat Weight", 
       y = "Price", 
       col = "Carat Weight Binned") + 
  ggtitle("Relationship between Carat Weight and Price") + 
  theme_bw()

# Not only visually can we see that there is heteroskedasticity, we can calculate 
# the standard deviation and see that its larger for diamonds weighing greather than 2 carats
diamond_dat %>% 
  group_by(carat_weight_binned) %>% 
  summarize(std_dev = sd(price))

# typically heteroskedasticity at larger values means it is long-tailed (mostly smaller 
# values and some really large ones). A good transformation for long-tailed data is 
# the natural log transformation. Above in the mutate step we calculated the log transform. 
# If we re-plot the data using the log price, then we can see that heteroskedasticity 
# issue is now fixed.

ggplot(data=diamond_dat, mapping=aes(x=carat_weight, y=log_price)) +
  geom_point(aes(color=carat_weight_binned)) + 
  labs(x = "Carat Weight", 
       y = "Log Price", 
       col = "Carat Weight Binned") + 
  ggtitle("Relationship between Carat Weight and Log Price") + 
  theme_bw()

# However, looking at this plot we see that the relationship is no longer linear. 
# In some sense we've fixed the heteroskedasticity, but created a non-linear relationship. 
# If we check the first plot, it appeared linear, so why don't we transform carat 
# with the log transform the same way we did to price. This will keep our heteroskedasticity 
# fix, but then bring back the linear relationship. Let's look at the plot:

ggplot(data=diamond_dat, mapping=aes(x=log_carat_weight, y=log_price)) +
  geom_point(aes(color=carat_weight_binned)) + 
  labs(x = "Log Carat Weight", 
       y = "Log Price", 
       col = "Carat Weight Binned") + 
  ggtitle("Relationship between Log Carat Weight and Log Price") + 
  theme_bw()

# Now that we've got some transformed variables and we are confident in the linearity 
# assumption, we'll move on to modeling. Before we start though we will split the 
# dataset into two parts: 1) Train and 2) Test. The train dataset is used to build 
# the model. The test dataset is used to evaluate how well the model performs 
# against data that it has not seen before. This will give us a good sense for the 
# accuracy of our model when we deploy it against new data in the real world.

diamond_dat_train <- filter(diamond_dat, dataset == "Train")
diamond_dat_test <- filter(diamond_dat, dataset == "Test")


# building a regression model in R ---------------------------------------------

# In R the function called "lm()" will fit a linear model. Just like you do in StatTools 
# you will have to tell R which variables to use as the independent and dependent variables. 
# In this case, let's start simple with using the log carat weight to predict the log price. 
# In R you specify this relationship like this: log_price ~ log_carat_weight. 

my_regression_model <- lm(log_price ~ log_carat_weight, 
                          data = diamond_dat_train)
my_regression_model

# You'll notice that when you print the model to the screen it tells you the coefficients. 
# The intercept and the coefficient weight for log_carat_weight. This is a simple 
# overview of the model but you can get more statistics about your model. The 
# "summary()" function will tell you the p-values, R-squared, and more!
summary(my_regression_model)


# assumptions testing on the residuals -----------------------------------------

# Remember back in DA when you were told that linear regresssions have assumptions 
# and that you must check those assumptions to see if the model is valid? The assumptions 
# you need to test are that: 
#  1. The residuals are normally distributed
#  2. The residuals are independent (no pattern over time)
#  3. The residuals have constant variance

# R saves the residuals from your model so you can test all of these assumptions 
# quickly. Here are the first six residuals:
head(residuals(my_regression_model))

# In order to check if the residuals are normal, you must perform a Kolmogorov-Smirnov test. 
# If the p-value is significant, then it means that the residuals are not normally distributed
ks.test(x=my_regression_model$residuals, y="pnorm")

# Here we see that the p-value is very low meaning that the data are not normal. 
# If you plot the data it appears that they are close to being normal. This is common 
# in data analysis when you have thousands of records. It's easy to make something significant
# even if it's only a slight difference. We will proceed with the analysis assuming normality.
ggplot() + 
  geom_histogram(aes(my_regression_model$residuals))

# Another common technique is to plot the residuals across the fitted values to see 
# if the variance is increasing or decreasing over our predictions. 
ggplot() + 
  geom_point(aes(x = my_regression_model$fitted.values, 
                 y = my_regression_model$residuals))

# Here we see that they are relatively constant. Good enough to pass the test. 
# Finally, let's check to see if the residuals are independent.
ggplot() + 
  geom_point(aes(x = seq.int(length(my_regression_model$residuals)),
                 y = my_regression_model$residuals))

# It's hard to tell from the plot because there are so many points, but things appear 
# to not have a pattern across the chart, so we're good to go on testing independence.

# selecting variables in a regression model ------------------------------------

# You may remember that StatTools had a feature that performed "Stepwise" regression 
# where you specified a list of candidate variables and StatTools went through a 
# process to determine which variables were significant enough to include in your 
# model. R has the same process so you don't have to always tell it which variables 
# to include. 

stepwise_reg_model <- step(lm(log_price ~ log_carat_weight + cut + color + clarity + report, 
                              data = diamond_dat_train))

# Looking at the stepwise model it determined that all the variables we included were 
# significant. You can look at the p-values to confirm.
summary(stepwise_reg_model)

# A summary of what's been shown so far: 
#  - How to read in a CSV file
#  - How to create transformations on the data
#  - How to plot the data
#  - How to build a multiple linear regression model
#  - How to check the assumptions of a regression model
#  - How to build a stepwise linear regression model

# Regression is a great tool, but some people criticize it's ability to handle 
# more complex, non-linear relationships along with it being hard to understand 
# for people who don't know much about statistics. Let's check out a different type 
# of model called a Decision Tree.

# building a decision tree model -----------------------------------------------

# A decision tree predictive model is not exactly like the one you may have learned 
# about in DA class. In DA class that was talking about a logic tree that followed 
# through all the potential logical outcomes in order to help you chose the best one. 
# In machine learning a decision tree is a learned pattern from the data that shows 
# you the logic on how to predict a value given some inputs. 

# The function "rpart()" creates a tree for you by picking the best variables. The 
# function is very similar to the "lm()" function in that you tell it which variables 
# to consider by using the notation log_price ~ log_carat_weight + cut + color + clarity + report

my_decision_tree <- rpart(log_price ~ log_carat_weight + clarity + report, 
                          data = diamond_dat_train)

# Just like with the regression, if you use the "summary()" function then you will 
# see more details about the tree. This may seem complex at first, so after the 
# summary we will try to visualize what is happening by using "rpart.plot()"

summary(my_decision_tree)

rpart.plot(my_decision_tree)

# You can see in the output how the tree would make a prediction about the price. 
# At first it looks at the log_carat_weight and if it is less than .33 (~1.4 carats) 
# then it moves to the left side of the tree. From there it checks whether the carat 
# weight is less than -.005 (< 1 carat), if so, then it makes a simple prediction about 
# the log_price being equal to 8.3 (~ $4,000). Basically the tree found a patter for 
# us that says, if a diamond in our dataset is less than 1 carat, it's probably valued 
# at something close to $4,000. The insights get much richer for larger diamonds. 
# For example, if the diamond is less than 1.4 carats but greater than 1 carat the model 
# considers whether it has clarity=SI1 or VS2, if so then it's only valued at $6,000 
# instead of $9,000 when it has a better clarity than those. 

# Because a tree is easy to visualize and comes up with easy to understand complex 
# relationships people tend to use them more often. However, let's see how the 
# models compare by looking at the mean absolute percentage error (MAPE) on the test set. 


# comparing the regression and decision tree models ----------------------------

# First, let's build the models considering the same set of potential (candidate) variables 

reg_model <- step(lm(log_price ~ log_carat_weight + cut + color + clarity + report, 
                  data = diamond_dat_train), trace = 0)

tree_model <- rpart(log_price ~ log_carat_weight + cut + color + clarity + report, 
                    data = diamond_dat_train)

# Second, let's predict values of the test dataset using the models. The "predict()" 
# function in R will use the model to generate a prediction for you.

reg_predictions <- predict(reg_model, newdata=diamond_dat_test)
tree_predictions <- predict(tree_model, newdata=diamond_dat_test)

# You can get a sense for the difference just by looking at the first few 
head(reg_predictions)
head(tree_predictions)

# Before we proceed we need to convert the log prices back to real prices by 
# taking the exponent of the prediction
reg_predictions <- exp(reg_predictions)
tree_predictions <- exp(tree_predictions)

# Now let's calculate the MAPE for each model
reg_mape <- mean(abs(reg_predictions - diamond_dat_test$price) / diamond_dat_test$price)
message(sprintf("MAPE for the Regression Model: %1.2f%%", reg_mape*100))

tree_mape <- mean(abs(tree_predictions - diamond_dat_test$price) / diamond_dat_test$price)
message(sprintf("MAPE for the Decision Tree Model: %1.2f%%", tree_mape*100))

# It looks like the regression model won! It had a 7.77% MAPE while the tree had 
# a MAPE equal to 18.20%. 

# Congratulations! You have learned how to build two different predictive models, 
# make predictions, and figure which one is better to use. The process we've shown 
# here is very similar to the type of work that data scientists do on a daily basis. 

