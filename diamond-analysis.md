Sarah Gets a Diamond
================
Dang Trinh - April 11, 2018

-   [Environment Setup](#environment-setup)
-   [Data Wrangling](#data-wrangling)
-   [Data Visualization](#data-visualization)
-   [Building Predictive Models](#building-predictive-models)
    -   [Regression Models](#regression-models)
        -   [Backward Step-Wise Linear Regression](#backward-step-wise-linear-regression)
    -   [Tree-based models](#tree-based-models)
        -   [Single Tuned Tree](#single-tuned-tree)
        -   [Bagged Tree/Random Forest](#bagged-treerandom-forest)
        -   [Random Forest](#random-forest)
        -   [Boosted Trees](#boosted-trees)
        -   [Lasso Regression](#lasso-regression)
    -   [Ensemble Forecasts](#ensemble-forecasts)
-   [Summary of Analysis & Areas for Further Research](#summary-of-analysis-areas-for-further-research)

Environment Setup
=================

R is an open source programming language that allows users to extend R by writing "packages". These packages usually perform a very complex and specific set of functions that we would like to utilize in our script. For example, the **gbm** package allows us to fit boosted tree predictive models.

You can load a package by using the `library()` function as shown below, but you must first install the package. You can install a package by using the `install.packages()` function like so:

``` r
install.packages("gbm")
install.packages("here")
```

After installing all of the packages listed below, then load them up by running this block of code. It is always a good idea to load all the required packages at the beginning of your script so that others can know what packages they need to replicate your analysis.

``` r
options(scipen=999, digits=6)
library(here)
library(Lahman)
library(lubridate)
library(forecast)
library(dplyr)
library(tidyr)
library(zoo)
library(glmnet)
library(tseries)
library(rpart)
library(rpart.plot)
library(glmnet)
library(forecast)
library(MASS)
library(randomForest)
library(gbm)
#library(prophet)
#library(vars)
#library(tree)
```

``` r
# create a folder to save our analyses, underlying data
todays_date_formatted <- format(Sys.Date(), '%Y%m%d')
dir.create(here::here('output', todays_date_formatted), showWarnings = FALSE)
```

Data Wrangling
==============

Data is usually not in a format that is ready for analysis, so we go through a number of steps that shape the data into a format that R can use for running analysis. We first read in the data and explore the various variables. The dataset contains several key attributes of diamond: carat, cut, color, clarity, polish, and symmetry. The `summary()` gives a quick idea of what they data entails.

``` r
# read the data from github
data_url <- 'https://raw.githubusercontent.com/DardenDSC/sarah-gets-a-diamond/master/data/sarah-gets-a-diamond-raw-data.csv'
raw_diamond_dat <- read.csv(data_url)

# get a sense for what the data entails
summary(raw_diamond_dat)
```

    ##        ID        Carat.Weight               Cut       Color    Clarity    
    ##  Min.   :   1   Min.   :0.75   Fair           : 199   D:1007   FL  :   4  
    ##  1st Qu.:2286   1st Qu.:1.01   Good           :1081   E:1189   IF  : 311  
    ##  Median :4572   Median :1.13   Ideal          :3783   F:1535   SI1 :3110  
    ##  Mean   :4572   Mean   :1.34   Signature-Ideal: 375   G:2253   VS1 :1826  
    ##  3rd Qu.:6857   3rd Qu.:1.59   Very Good      :3704   H:1710   VS2 :2396  
    ##  Max.   :9142   Max.   :2.91                          I:1448   VVS1: 451  
    ##                                                                VVS2:1044  
    ##  Polish    Symmetry   Report         Price         Dataset    
    ##  EX:3704   EX:3146   AGSL:1119   Min.   :  2184   Test :3142  
    ##  G : 894   G :1404   GIA :8023   1st Qu.:  5195   Train:6000  
    ##  ID: 900   ID: 925               Median :  7868               
    ##  VG:3644   VG:3667               Mean   : 11799               
    ##                                  3rd Qu.: 15098               
    ##                                  Max.   :101561               
    ## 

To facilitate subsequent regressions, we will do a minor cleaning of the `cut` variable to remove the space and the hyphen in its values ("Signature-Ideal" and "Very Good"). We will also create a log transformation and a reciprocal of Carat Weight and several bins variables for Carat Weight. These bin values were determined based on a subsequent scatter plot between price and carat weight.

``` r
# create a new variable to keep the raw data separate
diamond <- raw_diamond_dat

diamond <- diamond %>%
  mutate(Clarity = as.factor(Clarity), 
         Dataset = as.factor(Dataset), 
         Cut = as.factor(ifelse(Cut == "Signature-Ideal",
                                "SignatureIdeal", 
                                as.character(Cut))),
         Cut = as.factor(ifelse(Cut == "Very Good", 
                                "VeryGood", 
                                as.character(Cut))),
         LPrice = log(Price),
         LCarat = log(Carat.Weight),
         recipCarat = 1 / Carat.Weight,
         Caratbelow1 = as.numeric(Carat.Weight < 1),
         Caratequal1 = as.numeric(Carat.Weight == 1),
         Caratbelow1.5 = as.numeric((Carat.Weight > 1) & (Carat.Weight < 1.5)),
         Caratequal1.5 = as.numeric(Carat.Weight == 1.5),
         Caratbelow2 = as.numeric((Carat.Weight > 1.5) & (Carat.Weight < 2)),
         Caratabove2 = as.numeric(Carat.Weight >= 2))

summary(diamond)
```

    ##        ID        Carat.Weight              Cut       Color    Clarity    
    ##  Min.   :   1   Min.   :0.75   Fair          : 199   D:1007   FL  :   4  
    ##  1st Qu.:2286   1st Qu.:1.01   Good          :1081   E:1189   IF  : 311  
    ##  Median :4572   Median :1.13   Ideal         :3783   F:1535   SI1 :3110  
    ##  Mean   :4572   Mean   :1.34   SignatureIdeal: 375   G:2253   VS1 :1826  
    ##  3rd Qu.:6857   3rd Qu.:1.59   VeryGood      :3704   H:1710   VS2 :2396  
    ##  Max.   :9142   Max.   :2.91                         I:1448   VVS1: 451  
    ##                                                               VVS2:1044  
    ##  Polish    Symmetry   Report         Price         Dataset    
    ##  EX:3704   EX:3146   AGSL:1119   Min.   :  2184   Test :3142  
    ##  G : 894   G :1404   GIA :8023   1st Qu.:  5195   Train:6000  
    ##  ID: 900   ID: 925               Median :  7868               
    ##  VG:3644   VG:3667               Mean   : 11799               
    ##                                  3rd Qu.: 15098               
    ##                                  Max.   :101561               
    ##                                                               
    ##      LPrice          LCarat           recipCarat     Caratbelow1   
    ##  Min.   : 7.69   Min.   :-0.28768   Min.   :0.344   Min.   :0.000  
    ##  1st Qu.: 8.56   1st Qu.: 0.00995   1st Qu.:0.629   1st Qu.:0.000  
    ##  Median : 8.97   Median : 0.12222   Median :0.885   Median :0.000  
    ##  Mean   : 9.10   Mean   : 0.23232   Mean   :0.834   Mean   :0.199  
    ##  3rd Qu.: 9.62   3rd Qu.: 0.46373   3rd Qu.:0.990   3rd Qu.:0.000  
    ##  Max.   :11.53   Max.   : 1.06815   Max.   :1.333   Max.   :1.000  
    ##                                                                    
    ##   Caratequal1    Caratbelow1.5   Caratequal1.5     Caratbelow2   
    ##  Min.   :0.000   Min.   :0.000   Min.   :0.0000   Min.   :0.000  
    ##  1st Qu.:0.000   1st Qu.:0.000   1st Qu.:0.0000   1st Qu.:0.000  
    ##  Median :0.000   Median :0.000   Median :0.0000   Median :0.000  
    ##  Mean   :0.049   Mean   :0.411   Mean   :0.0237   Mean   :0.103  
    ##  3rd Qu.:0.000   3rd Qu.:1.000   3rd Qu.:0.0000   3rd Qu.:0.000  
    ##  Max.   :1.000   Max.   :1.000   Max.   :1.0000   Max.   :1.000  
    ##                                                                  
    ##   Caratabove2   
    ##  Min.   :0.000  
    ##  1st Qu.:0.000  
    ##  Median :0.000  
    ##  Mean   :0.214  
    ##  3rd Qu.:0.000  
    ##  Max.   :1.000  
    ## 

Here we will create several dummy variables, interaction terms, and split the data into the training and test set. The process of creating additional variables for analysis is typically referred to as "feature engineering". Feature engineering helps to find more nuanced relationships and usually leads to improved accuracy in predictive models

``` r
dummies <- model.matrix(~ 0 + Cut + Color + Clarity + Polish + Symmetry + Report + 
                              Cut:Color + Cut:Clarity + Cut:Polish + Cut:Symmetry + Cut:Report +
                              Color:Clarity + Color:Polish + Color:Symmetry + Color:Report+
                              Polish:Symmetry + Polish:Report + Symmetry:Report, 
                        data = diamond)

diamond.full <- as.data.frame(cbind(diamond, dummies))

diamond.train <- diamond[diamond$Dataset == "Train",]
diamond.test <- diamond[diamond$Dataset == "Test",]

diamond.full.train <- diamond.full[diamond.full$Dataset == "Train",]
diamond.full.test <- diamond.full[diamond.full$Dataset == "Test",]
```

We will also split the data into a smaller training set and a validation set.

``` r
nTrain <- dim(diamond.train)[1]
(nSmallTrain <- round(nrow(diamond.train) * 0.75))
```

    ## [1] 4500

``` r
(nValid <- nTrain - nSmallTrain)
```

    ## [1] 1500

``` r
rowIndicesSmallerTrain <- sample(1:nTrain, size = nSmallTrain, replace = FALSE)

diamond.smaller.train <- diamond.train[rowIndicesSmallerTrain, ]
diamond.validation <- diamond.train[-rowIndicesSmallerTrain, ]

diamond.full.smaller.train <- diamond.full.train[rowIndicesSmallerTrain, ]
diamond.full.validation <- diamond.full.train[-rowIndicesSmallerTrain, ]
```

Data Visualization
==================

An initial scatterplot of the Price vs. Carat Weight shows that Carat Weight typically falls into distinct buckets, and that there are significant heteroskedasticity in the relationship.

``` r
plot(x=diamond$Carat.Weight, y=diamond$Price, 
     main="Price vs. Carat", 
     ylab="Price", xlab="Carat")
```

![](diamond-analysis_files/figure-markdown_github/plot-carat-and-price-1.png)

Our second scatterplot of the Log Price vs. Carat Weight shows a quadratic relationship. Furthermore, the heteroskedasticity issue is now fixed.

``` r
plot(x=diamond$Carat.Weight, y=diamond$LPrice, 
     main="Log Price vs. Carat", 
     ylab="Log Price", xlab="Carat")
```

![](diamond-analysis_files/figure-markdown_github/plot-carat-and-log-price-1.png)

Finally, our last scatterplot of Log Price vs. Log Carat Weight shows a linear relationship with little heteroskedasticity. We will adopt this equation for our model. For more visualization, see the posted Tableau file.

Note that I currently encounter the IOPub data rate exceeded error below so the chart does not show up in the output. The tableau file does have the chart though.

``` r
plot(x=diamond$LCarat, y=diamond$LPrice, 
     main="Log Price vs. Log Carat", 
     ylab="Log Price", xlab="Log Carat")
```

![](diamond-analysis_files/figure-markdown_github/plot-log-carat-and-log-price-1.png)

Building Predictive Models
==========================

Regression Models
-----------------

Another class of model we could use is linear regression model. In this section we will use several approaches to calibrate our linear regression model.

### Backward Step-Wise Linear Regression

We start by including all categorical variables & possible interactions in our linear model. This yields a MAPE of 5.34222% on the validation set.

``` r
lm_formula <- "LPrice ~ LCarat+recipCarat+Caratbelow1+Caratequal1+Caratbelow1.5+Caratequal1.5+Caratbelow2+Caratabove2+CutFair+CutGood+CutIdeal+CutSignatureIdeal+CutVeryGood+ColorE+ColorF+ColorG+ColorH+ColorI+ClarityIF+ClaritySI1+ClarityVS1+ClarityVS2+ClarityVVS1+ClarityVVS2+PolishG+PolishID+PolishVG+SymmetryG+SymmetryID+SymmetryVG+ReportGIA+CutGood:ColorE+CutIdeal:ColorE+CutSignatureIdeal:ColorE+CutVeryGood:ColorE+CutGood:ColorF+CutIdeal:ColorF+CutSignatureIdeal:ColorF+CutVeryGood:ColorF+CutGood:ColorG+CutIdeal:ColorG+CutSignatureIdeal:ColorG+CutVeryGood:ColorG+CutGood:ColorH+CutIdeal:ColorH+CutSignatureIdeal:ColorH+CutVeryGood:ColorH+CutGood:ColorI+CutIdeal:ColorI+CutSignatureIdeal:ColorI+CutVeryGood:ColorI+CutGood:ClarityIF+CutIdeal:ClarityIF+CutSignatureIdeal:ClarityIF+CutVeryGood:ClarityIF+CutGood:ClaritySI1+CutIdeal:ClaritySI1+CutSignatureIdeal:ClaritySI1+CutVeryGood:ClaritySI1+CutGood:ClarityVS1+CutIdeal:ClarityVS1+CutSignatureIdeal:ClarityVS1+CutVeryGood:ClarityVS1+CutGood:ClarityVS2+CutIdeal:ClarityVS2+CutSignatureIdeal:ClarityVS2+CutVeryGood:ClarityVS2+CutGood:ClarityVVS1+CutIdeal:ClarityVVS1+CutSignatureIdeal:ClarityVVS1+CutVeryGood:ClarityVVS1+CutGood:ClarityVVS2+CutIdeal:ClarityVVS2+CutSignatureIdeal:ClarityVVS2+CutVeryGood:ClarityVVS2+CutGood:PolishG+CutIdeal:PolishG+CutSignatureIdeal:PolishG+CutVeryGood:PolishG+CutGood:PolishID+CutIdeal:PolishID+CutSignatureIdeal:PolishID+CutVeryGood:PolishID+CutGood:PolishVG+CutIdeal:PolishVG+CutSignatureIdeal:PolishVG+CutVeryGood:PolishVG+CutGood:SymmetryG+CutIdeal:SymmetryG+CutSignatureIdeal:SymmetryG+CutVeryGood:SymmetryG+CutGood:SymmetryID+CutIdeal:SymmetryID+CutSignatureIdeal:SymmetryID+CutVeryGood:SymmetryID+CutGood:SymmetryVG+CutIdeal:SymmetryVG+CutSignatureIdeal:SymmetryVG+CutVeryGood:SymmetryVG+CutGood:ReportGIA+CutIdeal:ReportGIA+CutSignatureIdeal:ReportGIA+CutVeryGood:ReportGIA+ColorE:ClarityIF+ColorF:ClarityIF+ColorG:ClarityIF+ColorH:ClarityIF+ColorI:ClarityIF+ColorE:ClaritySI1+ColorF:ClaritySI1+ColorG:ClaritySI1+ColorH:ClaritySI1+ColorI:ClaritySI1+ColorE:ClarityVS1+ColorF:ClarityVS1+ColorG:ClarityVS1+ColorH:ClarityVS1+ColorI:ClarityVS1+ColorE:ClarityVS2+ColorF:ClarityVS2+ColorG:ClarityVS2+ColorH:ClarityVS2+ColorI:ClarityVS2+ColorE:ClarityVVS1+ColorF:ClarityVVS1+ColorG:ClarityVVS1+ColorH:ClarityVVS1+ColorI:ClarityVVS1+ColorE:ClarityVVS2+ColorF:ClarityVVS2+ColorG:ClarityVVS2+ColorH:ClarityVVS2+ColorI:ClarityVVS2+ColorE:PolishG+ColorF:PolishG+ColorG:PolishG+ColorH:PolishG+ColorI:PolishG+ColorE:PolishID+ColorF:PolishID+ColorG:PolishID+ColorH:PolishID+ColorI:PolishID+ColorE:PolishVG+ColorF:PolishVG+ColorG:PolishVG+ColorH:PolishVG+ColorI:PolishVG+ColorE:SymmetryG+ColorF:SymmetryG+ColorG:SymmetryG+ColorH:SymmetryG+ColorI:SymmetryG+ColorE:SymmetryID+ColorF:SymmetryID+ColorG:SymmetryID+ColorH:SymmetryID+ColorI:SymmetryID+ColorE:SymmetryVG+ColorF:SymmetryVG+ColorG:SymmetryVG+ColorH:SymmetryVG+ColorI:SymmetryVG+ColorE:ReportGIA+ColorF:ReportGIA+ColorG:ReportGIA+ColorH:ReportGIA+ColorI:ReportGIA+PolishG:SymmetryG+PolishID:SymmetryG+PolishVG:SymmetryG+PolishG:SymmetryID+PolishID:SymmetryID+PolishVG:SymmetryID+PolishG:SymmetryVG+PolishID:SymmetryVG+PolishVG:SymmetryVG+PolishG:ReportGIA+PolishID:ReportGIA+PolishVG:ReportGIA+SymmetryG:ReportGIA+SymmetryID:ReportGIA+SymmetryVG:ReportGIA
          + LCarat:Cut + LCarat:Color + LCarat:Calarity + LCarat:Polish + LCarat:Symmetry + LCarat:Report
          + Caratbelow1:Cut + Caratbelow1:Color + Caratbelow1:Calarity + 
            Caratbelow1:Polish + Caratbelow1:Symmetry + Caratbelow1:Report"
```

``` r
lm <- lm(as.formula(lm_formula), data = diamond.full.smaller.train)
summary(lm)
```

    ## 
    ## Call:
    ## lm(formula = as.formula(lm_formula), data = diamond.full.smaller.train)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -0.74631 -0.04303 -0.00119  0.04119  0.42663 
    ## 
    ## Coefficients: (26 not defined because of singularities)
    ##                                 Estimate Std. Error t value
    ## (Intercept)                   10.8655590  0.0980612 110.804
    ## LCarat                         0.8351305  0.0507200  16.465
    ## recipCarat                    -1.0238051  0.0584259 -17.523
    ## Caratbelow1                   -0.3489892  0.0135394 -25.776
    ## Caratequal1                   -0.2604478  0.0131872 -19.750
    ## Caratbelow1.5                 -0.3024472  0.0114697 -26.369
    ## Caratequal1.5                 -0.2254955  0.0104495 -21.579
    ## Caratbelow2                   -0.2219400  0.0076067 -29.177
    ## Caratabove2                           NA         NA      NA
    ## CutFair                        0.0202809  0.0530545   0.382
    ## CutGood                       -0.0622608  0.0287534  -2.165
    ## CutIdeal                       0.0203315  0.0798551   0.255
    ## CutSignatureIdeal              0.2305907  0.0425615   5.418
    ## CutVeryGood                           NA         NA      NA
    ## ColorE                        -0.0523105  0.0593955  -0.881
    ## ColorF                        -0.1891978  0.0580192  -3.261
    ## ColorG                        -0.3330409  0.0579257  -5.749
    ## ColorH                        -0.5337221  0.0567669  -9.402
    ## ColorI                        -0.6193065  0.0601847 -10.290
    ## ClarityIF                      0.0020389  0.0908589   0.022
    ## ClaritySI1                    -0.9762597  0.0688067 -14.188
    ## ClarityVS1                    -0.7155176  0.0723355  -9.892
    ## ClarityVS2                    -0.8177066  0.0691154 -11.831
    ## ClarityVVS1                   -0.2546515  0.0596331  -4.270
    ## ClarityVVS2                   -0.4241325  0.0574438  -7.383
    ## PolishG                       -0.0910578  0.0470799  -1.934
    ## PolishID                       0.0825737  0.0843585   0.979
    ## PolishVG                      -0.0243705  0.0298757  -0.816
    ## SymmetryG                     -0.0770043  0.0396723  -1.941
    ## SymmetryID                     0.0080318  0.0854160   0.094
    ## SymmetryVG                    -0.0585867  0.0342242  -1.712
    ## ReportGIA                      0.0779037  0.0527068   1.478
    ## CutGood:ColorE                 0.0392831  0.0364727   1.077
    ## CutIdeal:ColorE                0.0315464  0.0357619   0.882
    ## CutSignatureIdeal:ColorE      -0.0035512  0.0442330  -0.080
    ## CutVeryGood:ColorE             0.0364012  0.0351502   1.036
    ## CutGood:ColorF                 0.0241971  0.0389294   0.622
    ## CutIdeal:ColorF                0.0287357  0.0381122   0.754
    ## CutSignatureIdeal:ColorF      -0.0171448  0.0456438  -0.376
    ## CutVeryGood:ColorF             0.0293738  0.0375618   0.782
    ## CutGood:ColorG                 0.0628889  0.0399958   1.572
    ## CutIdeal:ColorG                0.0414023  0.0393945   1.051
    ## CutSignatureIdeal:ColorG      -0.0036291  0.0458689  -0.079
    ## CutVeryGood:ColorG             0.0468568  0.0388741   1.205
    ## CutGood:ColorH                 0.0818769  0.0385186   2.126
    ## CutIdeal:ColorH                0.0731451  0.0377812   1.936
    ## CutSignatureIdeal:ColorH       0.0438032  0.0452115   0.969
    ## CutVeryGood:ColorH             0.0738717  0.0372394   1.984
    ## CutGood:ColorI                -0.0019963  0.0412084  -0.048
    ## CutIdeal:ColorI               -0.0356776  0.0405920  -0.879
    ## CutSignatureIdeal:ColorI      -0.0651230  0.0474816  -1.372
    ## CutVeryGood:ColorI            -0.0207863  0.0399651  -0.520
    ## CutGood:ClarityIF              0.0108352  0.0739743   0.146
    ## CutIdeal:ClarityIF             0.0061866  0.1030868   0.060
    ## CutSignatureIdeal:ClarityIF   -0.0740820  0.0777328  -0.953
    ## CutVeryGood:ClarityIF         -0.0274904  0.0702934  -0.391
    ## CutGood:ClaritySI1             0.0336482  0.0411050   0.819
    ## CutIdeal:ClaritySI1            0.0189643  0.0851220   0.223
    ## CutSignatureIdeal:ClaritySI1   0.0007003  0.0428411   0.016
    ## CutVeryGood:ClaritySI1         0.0013850  0.0390805   0.035
    ## CutGood:ClarityVS1             0.0834733  0.0471883   1.769
    ## CutIdeal:ClarityVS1            0.0938302  0.0878981   1.067
    ## CutSignatureIdeal:ClarityVS1   0.0558192  0.0476801   1.171
    ## CutVeryGood:ClarityVS1         0.0734765  0.0449613   1.634
    ## CutGood:ClarityVS2             0.0395171  0.0419231   0.943
    ## CutIdeal:ClarityVS2            0.0451698  0.0853779   0.529
    ## CutSignatureIdeal:ClarityVS2   0.0198754  0.0432865   0.459
    ## CutVeryGood:ClarityVS2         0.0246924  0.0398227   0.620
    ## CutGood:ClarityVVS1           -0.0189765  0.0342842  -0.554
    ## CutIdeal:ClarityVVS1           0.0005402  0.0767547   0.007
    ## CutSignatureIdeal:ClarityVVS1 -0.0281100  0.0262738  -1.070
    ## CutVeryGood:ClarityVVS1               NA         NA      NA
    ## CutGood:ClarityVVS2                   NA         NA      NA
    ## CutIdeal:ClarityVVS2           0.0234357  0.0758484   0.309
    ## CutSignatureIdeal:ClarityVVS2         NA         NA      NA
    ## CutVeryGood:ClarityVVS2               NA         NA      NA
    ## CutGood:PolishG                0.0330801  0.0278699   1.187
    ## CutIdeal:PolishG              -0.0190586  0.0299881  -0.636
    ## CutSignatureIdeal:PolishG             NA         NA      NA
    ## CutVeryGood:PolishG            0.0033709  0.0269098   0.125
    ## CutGood:PolishID                      NA         NA      NA
    ## CutIdeal:PolishID              0.0562382  0.0670207   0.839
    ## CutSignatureIdeal:PolishID    -0.0065754  0.0378993  -0.173
    ## CutVeryGood:PolishID                  NA         NA      NA
    ## CutGood:PolishVG               0.0095665  0.0218698   0.437
    ## CutIdeal:PolishVG             -0.0076598  0.0212572  -0.360
    ## CutSignatureIdeal:PolishVG            NA         NA      NA
    ## CutVeryGood:PolishVG          -0.0003595  0.0209466  -0.017
    ## CutGood:SymmetryG              0.0078009  0.0308538   0.253
    ## CutIdeal:SymmetryG             0.0393989  0.0311091   1.266
    ## CutSignatureIdeal:SymmetryG           NA         NA      NA
    ## CutVeryGood:SymmetryG          0.0255379  0.0287914   0.887
    ## CutGood:SymmetryID            -0.0121378  0.0833795  -0.146
    ## CutIdeal:SymmetryID           -0.0477363  0.0693373  -0.688
    ## CutSignatureIdeal:SymmetryID          NA         NA      NA
    ## CutVeryGood:SymmetryID                NA         NA      NA
    ## CutGood:SymmetryVG            -0.0088170  0.0301987  -0.292
    ## CutIdeal:SymmetryVG            0.0036635  0.0282153   0.130
    ## CutSignatureIdeal:SymmetryVG          NA         NA      NA
    ## CutVeryGood:SymmetryVG         0.0177055  0.0281748   0.628
    ## CutGood:ReportGIA              0.0236163  0.0361656   0.653
    ## CutIdeal:ReportGIA             0.0272885  0.0384694   0.709
    ## CutSignatureIdeal:ReportGIA           NA         NA      NA
    ## CutVeryGood:ReportGIA          0.0206604  0.0323810   0.638
    ## ColorE:ClarityIF              -0.1588139  0.0269057  -5.903
    ## ColorF:ClarityIF              -0.1873876  0.0230519  -8.129
    ## ColorG:ClarityIF              -0.3143946  0.0190301 -16.521
    ## ColorH:ClarityIF              -0.3011539  0.0262590 -11.469
    ## ColorI:ClarityIF              -0.3396678  0.0252712 -13.441
    ## ColorE:ClaritySI1              0.0575091  0.0148255   3.879
    ## ColorF:ClaritySI1              0.0932205  0.0141341   6.595
    ## ColorG:ClaritySI1              0.1752436  0.0132845  13.192
    ## ColorH:ClaritySI1              0.3284261  0.0145254  22.610
    ## ColorI:ClaritySI1              0.3571339  0.0162687  21.952
    ## ColorE:ClarityVS1              0.0377511  0.0174330   2.166
    ## ColorF:ClarityVS1              0.0811499  0.0162016   5.009
    ## ColorG:ClarityVS1              0.1154731  0.0149129   7.743
    ## ColorH:ClarityVS1              0.1593728  0.0164882   9.666
    ## ColorI:ClarityVS1              0.1532027  0.0180141   8.505
    ## ColorE:ClarityVS2              0.0768033  0.0160460   4.786
    ## ColorF:ClarityVS2              0.1322410  0.0150549   8.784
    ## ColorG:ClarityVS2              0.1899883  0.0138529  13.715
    ## ColorH:ClarityVS2              0.2546965  0.0154144  16.523
    ## ColorI:ClarityVS2              0.2467475  0.0170246  14.494
    ## ColorE:ClarityVVS1            -0.0142941  0.0250600  -0.570
    ## ColorF:ClarityVVS1            -0.0469289  0.0231930  -2.023
    ## ColorG:ClarityVVS1            -0.1029790  0.0213035  -4.834
    ## ColorH:ClarityVVS1            -0.1026783  0.0243161  -4.223
    ## ColorI:ClarityVVS1            -0.0967651  0.0280331  -3.452
    ## ColorE:ClarityVVS2                    NA         NA      NA
    ## ColorF:ClarityVVS2                    NA         NA      NA
    ## ColorG:ClarityVVS2                    NA         NA      NA
    ## ColorH:ClarityVVS2                    NA         NA      NA
    ## ColorI:ClarityVVS2                    NA         NA      NA
    ## ColorE:PolishG                -0.0128534  0.0177674  -0.723
    ## ColorF:PolishG                -0.0204235  0.0169001  -1.208
    ## ColorG:PolishG                -0.0032700  0.0164877  -0.198
    ## ColorH:PolishG                 0.0085675  0.0173562   0.494
    ## ColorI:PolishG                 0.0051852  0.0175001   0.296
    ## ColorE:PolishID               -0.1523420  0.1222093  -1.247
    ## ColorF:PolishID               -0.0640203  0.0822802  -0.778
    ## ColorG:PolishID               -0.0721611  0.0845047  -0.854
    ## ColorH:PolishID               -0.0786020  0.0799806  -0.983
    ## ColorI:PolishID               -0.0628388  0.0919282  -0.684
    ## ColorE:PolishVG                0.0001190  0.0110446   0.011
    ## ColorF:PolishVG               -0.0128382  0.0106159  -1.209
    ## ColorG:PolishVG                0.0010826  0.0098505   0.110
    ## ColorH:PolishVG               -0.0046653  0.0104302  -0.447
    ## ColorI:PolishVG                0.0038927  0.0110129   0.353
    ## ColorE:SymmetryG              -0.0086822  0.0159796  -0.543
    ## ColorF:SymmetryG               0.0017816  0.0159098   0.112
    ## ColorG:SymmetryG               0.0012565  0.0149571   0.084
    ## ColorH:SymmetryG               0.0141701  0.0155856   0.909
    ## ColorI:SymmetryG               0.0075375  0.0164451   0.458
    ## ColorE:SymmetryID              0.0619593  0.1314593   0.471
    ## ColorF:SymmetryID              0.0320174  0.0928084   0.345
    ## ColorG:SymmetryID              0.0266490  0.0944560   0.282
    ## ColorH:SymmetryID             -0.0093973  0.0912887  -0.103
    ## ColorI:SymmetryID              0.0221361  0.1013422   0.218
    ## ColorE:SymmetryVG              0.0086507  0.0115386   0.750
    ## ColorF:SymmetryVG              0.0220604  0.0111655   1.976
    ## ColorG:SymmetryVG              0.0164440  0.0102778   1.600
    ## ColorH:SymmetryVG              0.0319165  0.0109119   2.925
    ## ColorI:SymmetryVG              0.0159474  0.0115790   1.377
    ## ColorE:ReportGIA              -0.0974689  0.0532428  -1.831
    ## ColorF:ReportGIA              -0.0298037  0.0490084  -0.608
    ## ColorG:ReportGIA              -0.0465743  0.0480351  -0.970
    ## ColorH:ReportGIA              -0.0956322  0.0489295  -1.954
    ## ColorI:ReportGIA              -0.0586886  0.0485829  -1.208
    ## PolishG:SymmetryG              0.0381405  0.0182753   2.087
    ## PolishID:SymmetryG                    NA         NA      NA
    ## PolishVG:SymmetryG             0.0170066  0.0109906   1.547
    ## PolishG:SymmetryID                    NA         NA      NA
    ## PolishID:SymmetryID           -0.0243087  0.0438049  -0.555
    ## PolishVG:SymmetryID                   NA         NA      NA
    ## PolishG:SymmetryVG             0.0329921  0.0164350   2.007
    ## PolishID:SymmetryVG                   NA         NA      NA
    ## PolishVG:SymmetryVG            0.0232712  0.0062726   3.710
    ## PolishG:ReportGIA              0.0305191  0.0349096   0.874
    ## PolishID:ReportGIA                    NA         NA      NA
    ## PolishVG:ReportGIA            -0.0047860  0.0222122  -0.215
    ## SymmetryG:ReportGIA            0.0141380  0.0270330   0.523
    ## SymmetryID:ReportGIA                  NA         NA      NA
    ## SymmetryVG:ReportGIA           0.0013887  0.0214707   0.065
    ##                                           Pr(>|t|)    
    ## (Intercept)                   < 0.0000000000000002 ***
    ## LCarat                        < 0.0000000000000002 ***
    ## recipCarat                    < 0.0000000000000002 ***
    ## Caratbelow1                   < 0.0000000000000002 ***
    ## Caratequal1                   < 0.0000000000000002 ***
    ## Caratbelow1.5                 < 0.0000000000000002 ***
    ## Caratequal1.5                 < 0.0000000000000002 ***
    ## Caratbelow2                   < 0.0000000000000002 ***
    ## Caratabove2                                     NA    
    ## CutFair                                   0.702283    
    ## CutGood                                   0.030416 *  
    ## CutIdeal                                  0.799041    
    ## CutSignatureIdeal              0.00000006359526618 ***
    ## CutVeryGood                                     NA    
    ## ColorE                                    0.378521    
    ## ColorF                                    0.001119 ** 
    ## ColorG                         0.00000000956769078 ***
    ## ColorH                        < 0.0000000000000002 ***
    ## ColorI                        < 0.0000000000000002 ***
    ## ClarityIF                                 0.982097    
    ## ClaritySI1                    < 0.0000000000000002 ***
    ## ClarityVS1                    < 0.0000000000000002 ***
    ## ClarityVS2                    < 0.0000000000000002 ***
    ## ClarityVVS1                    0.00001993797156283 ***
    ## ClarityVVS2                    0.00000000000018393 ***
    ## PolishG                                   0.053164 .  
    ## PolishID                                  0.327712    
    ## PolishVG                                  0.414700    
    ## SymmetryG                                 0.052322 .  
    ## SymmetryID                                0.925089    
    ## SymmetryVG                                0.086995 .  
    ## ReportGIA                                 0.139465    
    ## CutGood:ColorE                            0.281515    
    ## CutIdeal:ColorE                           0.377759    
    ## CutSignatureIdeal:ColorE                  0.936015    
    ## CutVeryGood:ColorE                        0.300452    
    ## CutGood:ColorF                            0.534261    
    ## CutIdeal:ColorF                           0.450903    
    ## CutSignatureIdeal:ColorF                  0.707216    
    ## CutVeryGood:ColorF                        0.434250    
    ## CutGood:ColorG                            0.115934    
    ## CutIdeal:ColorG                           0.293332    
    ## CutSignatureIdeal:ColorG                  0.936941    
    ## CutVeryGood:ColorG                        0.228135    
    ## CutGood:ColorH                            0.033589 *  
    ## CutIdeal:ColorH                           0.052930 .  
    ## CutSignatureIdeal:ColorH                  0.332674    
    ## CutVeryGood:ColorH                        0.047353 *  
    ## CutGood:ColorI                            0.961364    
    ## CutIdeal:ColorI                           0.379488    
    ## CutSignatureIdeal:ColorI                  0.170276    
    ## CutVeryGood:ColorI                        0.603012    
    ## CutGood:ClarityIF                         0.883555    
    ## CutIdeal:ClarityIF                        0.952148    
    ## CutSignatureIdeal:ClarityIF               0.340626    
    ## CutVeryGood:ClarityIF                     0.695757    
    ## CutGood:ClaritySI1                        0.413064    
    ## CutIdeal:ClaritySI1                       0.823709    
    ## CutSignatureIdeal:ClaritySI1              0.986959    
    ## CutVeryGood:ClaritySI1                    0.971730    
    ## CutGood:ClarityVS1                        0.076974 .  
    ## CutIdeal:ClarityVS1                       0.285810    
    ## CutSignatureIdeal:ClarityVS1              0.241783    
    ## CutVeryGood:ClarityVS1                    0.102286    
    ## CutGood:ClarityVS2                        0.345933    
    ## CutIdeal:ClarityVS2                       0.596793    
    ## CutSignatureIdeal:ClarityVS2              0.646142    
    ## CutVeryGood:ClarityVS2                    0.535252    
    ## CutGood:ClarityVVS1                       0.579946    
    ## CutIdeal:ClarityVVS1                      0.994385    
    ## CutSignatureIdeal:ClarityVVS1             0.284730    
    ## CutVeryGood:ClarityVVS1                         NA    
    ## CutGood:ClarityVVS2                             NA    
    ## CutIdeal:ClarityVVS2                      0.757351    
    ## CutSignatureIdeal:ClarityVVS2                   NA    
    ## CutVeryGood:ClarityVVS2                         NA    
    ## CutGood:PolishG                           0.235314    
    ## CutIdeal:PolishG                          0.525112    
    ## CutSignatureIdeal:PolishG                       NA    
    ## CutVeryGood:PolishG                       0.900317    
    ## CutGood:PolishID                                NA    
    ## CutIdeal:PolishID                         0.401450    
    ## CutSignatureIdeal:PolishID                0.862269    
    ## CutVeryGood:PolishID                            NA    
    ## CutGood:PolishVG                          0.661822    
    ## CutIdeal:PolishVG                         0.718610    
    ## CutSignatureIdeal:PolishVG                      NA    
    ## CutVeryGood:PolishVG                      0.986306    
    ## CutGood:SymmetryG                         0.800408    
    ## CutIdeal:SymmetryG                        0.205410    
    ## CutSignatureIdeal:SymmetryG                     NA    
    ## CutVeryGood:SymmetryG                     0.375129    
    ## CutGood:SymmetryID                        0.884265    
    ## CutIdeal:SymmetryID                       0.491197    
    ## CutSignatureIdeal:SymmetryID                    NA    
    ## CutVeryGood:SymmetryID                          NA    
    ## CutGood:SymmetryVG                        0.770326    
    ## CutIdeal:SymmetryVG                       0.896699    
    ## CutSignatureIdeal:SymmetryVG                    NA    
    ## CutVeryGood:SymmetryVG                    0.529765    
    ## CutGood:ReportGIA                         0.513789    
    ## CutIdeal:ReportGIA                        0.478141    
    ## CutSignatureIdeal:ReportGIA                     NA    
    ## CutVeryGood:ReportGIA                     0.523480    
    ## ColorE:ClarityIF               0.00000000385063930 ***
    ## ColorF:ClarityIF               0.00000000000000056 ***
    ## ColorG:ClarityIF              < 0.0000000000000002 ***
    ## ColorH:ClarityIF              < 0.0000000000000002 ***
    ## ColorI:ClarityIF              < 0.0000000000000002 ***
    ## ColorE:ClaritySI1                         0.000106 ***
    ## ColorF:ClaritySI1              0.00000000004748691 ***
    ## ColorG:ClaritySI1             < 0.0000000000000002 ***
    ## ColorH:ClaritySI1             < 0.0000000000000002 ***
    ## ColorI:ClaritySI1             < 0.0000000000000002 ***
    ## ColorE:ClarityVS1                         0.030404 *  
    ## ColorF:ClarityVS1              0.00000056957729604 ***
    ## ColorG:ClarityVS1              0.00000000000001198 ***
    ## ColorH:ClarityVS1             < 0.0000000000000002 ***
    ## ColorI:ClarityVS1             < 0.0000000000000002 ***
    ## ColorE:ClarityVS2              0.00000175410524323 ***
    ## ColorF:ClarityVS2             < 0.0000000000000002 ***
    ## ColorG:ClarityVS2             < 0.0000000000000002 ***
    ## ColorH:ClarityVS2             < 0.0000000000000002 ***
    ## ColorI:ClarityVS2             < 0.0000000000000002 ***
    ## ColorE:ClarityVVS1                        0.568439    
    ## ColorF:ClarityVVS1                        0.043092 *  
    ## ColorG:ClarityVVS1             0.00000138512738513 ***
    ## ColorH:ClarityVVS1             0.00002464034211583 ***
    ## ColorI:ClarityVVS1                        0.000562 ***
    ## ColorE:ClarityVVS2                              NA    
    ## ColorF:ClarityVVS2                              NA    
    ## ColorG:ClarityVVS2                              NA    
    ## ColorH:ClarityVVS2                              NA    
    ## ColorI:ClarityVVS2                              NA    
    ## ColorE:PolishG                            0.469457    
    ## ColorF:PolishG                            0.226928    
    ## ColorG:PolishG                            0.842798    
    ## ColorH:PolishG                            0.621595    
    ## ColorI:PolishG                            0.767017    
    ## ColorE:PolishID                           0.212624    
    ## ColorF:PolishID                           0.436567    
    ## ColorG:PolishID                           0.393191    
    ## ColorH:PolishID                           0.325778    
    ## ColorI:PolishID                           0.494287    
    ## ColorE:PolishVG                           0.991403    
    ## ColorF:PolishVG                           0.226601    
    ## ColorG:PolishVG                           0.912495    
    ## ColorH:PolishVG                           0.654691    
    ## ColorI:PolishVG                           0.723752    
    ## ColorE:SymmetryG                          0.586929    
    ## ColorF:SymmetryG                          0.910845    
    ## ColorG:SymmetryG                          0.933054    
    ## ColorH:SymmetryG                          0.363304    
    ## ColorI:SymmetryG                          0.646728    
    ## ColorE:SymmetryID                         0.637437    
    ## ColorF:SymmetryID                         0.730123    
    ## ColorG:SymmetryID                         0.777856    
    ## ColorH:SymmetryID                         0.918015    
    ## ColorI:SymmetryID                         0.827105    
    ## ColorE:SymmetryVG                         0.453465    
    ## ColorF:SymmetryVG                         0.048244 *  
    ## ColorG:SymmetryVG                         0.109682    
    ## ColorH:SymmetryVG                         0.003463 ** 
    ## ColorI:SymmetryVG                         0.168499    
    ## ColorE:ReportGIA                          0.067221 .  
    ## ColorF:ReportGIA                          0.543129    
    ## ColorG:ReportGIA                          0.332305    
    ## ColorH:ReportGIA                          0.050707 .  
    ## ColorI:ReportGIA                          0.227109    
    ## PolishG:SymmetryG                         0.036947 *  
    ## PolishID:SymmetryG                              NA    
    ## PolishVG:SymmetryG                        0.121847    
    ## PolishG:SymmetryID                              NA    
    ## PolishID:SymmetryID                       0.578970    
    ## PolishVG:SymmetryID                             NA    
    ## PolishG:SymmetryVG                        0.044765 *  
    ## PolishID:SymmetryVG                             NA    
    ## PolishVG:SymmetryVG                       0.000210 ***
    ## PolishG:ReportGIA                         0.382040    
    ## PolishID:ReportGIA                              NA    
    ## PolishVG:ReportGIA                        0.829412    
    ## SymmetryG:ReportGIA                       0.601008    
    ## SymmetryID:ReportGIA                            NA    
    ## SymmetryVG:ReportGIA                      0.948434    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.07102 on 4342 degrees of freedom
    ## Multiple R-squared:  0.9905, Adjusted R-squared:  0.9901 
    ## F-statistic:  2870 on 157 and 4342 DF,  p-value: < 0.00000000000000022

``` r
lm.pred.valid <- predict(lm, diamond.full.validation)
accuracy(exp(lm.pred.valid), diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 40.44794 1148.506 661.4527 -0.5542636 5.512112

Here we will perform the step-wise backward regression by doing at most 10 steps to weed out the variables that are not considered significant. More steps may be needed to find the optimum model. The argument `trace=0` means that the diagnostics of each step are not printed to the screen.

``` r
lm.step <- step(lm, direction = "backward", trace=0, step=10)
lm.step.pred.valid <- predict(lm.step, diamond.full.validation)
accuracy(exp(lm.step.pred.valid), diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 40.44794 1148.506 661.4527 -0.5542636 5.512112

We will now perform the regression on the full training set.

``` r
formula <- "LPrice ~ LCarat+recipCarat+Caratbelow1+Caratequal1+Caratbelow1.5+Caratequal1.5+Caratbelow2+Caratabove2+CutFair+CutGood+CutIdeal+CutSignatureIdeal+CutVeryGood+ColorE+ColorF+ColorG+ColorH+ColorI+ClarityIF+ClaritySI1+ClarityVS1+ClarityVS2+ClarityVVS1+ClarityVVS2+PolishG+PolishID+PolishVG+SymmetryG+SymmetryID+SymmetryVG+ReportGIA+CutGood:ColorE+CutIdeal:ColorE+CutSignatureIdeal:ColorE+CutVeryGood:ColorE+CutGood:ColorF+CutIdeal:ColorF+CutSignatureIdeal:ColorF+CutVeryGood:ColorF+CutGood:ColorG+CutIdeal:ColorG+CutSignatureIdeal:ColorG+CutVeryGood:ColorG+CutGood:ColorH+CutIdeal:ColorH+CutSignatureIdeal:ColorH+CutVeryGood:ColorH+CutGood:ColorI+CutIdeal:ColorI+CutSignatureIdeal:ColorI+CutVeryGood:ColorI+CutGood:ClarityIF+CutIdeal:ClarityIF+CutSignatureIdeal:ClarityIF+CutVeryGood:ClarityIF+CutGood:ClaritySI1+CutIdeal:ClaritySI1+CutSignatureIdeal:ClaritySI1+CutVeryGood:ClaritySI1+CutGood:ClarityVS1+CutIdeal:ClarityVS1+CutSignatureIdeal:ClarityVS1+CutVeryGood:ClarityVS1+CutGood:ClarityVS2+CutIdeal:ClarityVS2+CutSignatureIdeal:ClarityVS2+CutVeryGood:ClarityVS2+CutGood:ClarityVVS1+CutIdeal:ClarityVVS1+CutSignatureIdeal:ClarityVVS1+CutVeryGood:ClarityVVS1+CutGood:ClarityVVS2+CutIdeal:ClarityVVS2+CutSignatureIdeal:ClarityVVS2+CutVeryGood:ClarityVVS2+CutGood:PolishG+CutIdeal:PolishG+CutSignatureIdeal:PolishG+CutVeryGood:PolishG+CutGood:PolishID+CutIdeal:PolishID+CutSignatureIdeal:PolishID+CutVeryGood:PolishID+CutGood:PolishVG+CutIdeal:PolishVG+CutSignatureIdeal:PolishVG+CutVeryGood:PolishVG+CutGood:SymmetryG+CutIdeal:SymmetryG+CutSignatureIdeal:SymmetryG+CutVeryGood:SymmetryG+CutGood:SymmetryID+CutIdeal:SymmetryID+CutSignatureIdeal:SymmetryID+CutVeryGood:SymmetryID+CutGood:SymmetryVG+CutIdeal:SymmetryVG+CutSignatureIdeal:SymmetryVG+CutVeryGood:SymmetryVG+CutGood:ReportGIA+CutIdeal:ReportGIA+CutSignatureIdeal:ReportGIA+CutVeryGood:ReportGIA+ColorE:ClarityIF+ColorF:ClarityIF+ColorG:ClarityIF+ColorH:ClarityIF+ColorI:ClarityIF+ColorE:ClaritySI1+ColorF:ClaritySI1+ColorG:ClaritySI1+ColorH:ClaritySI1+ColorI:ClaritySI1+ColorE:ClarityVS1+ColorF:ClarityVS1+ColorG:ClarityVS1+ColorH:ClarityVS1+ColorI:ClarityVS1+ColorE:ClarityVS2+ColorF:ClarityVS2+ColorG:ClarityVS2+ColorH:ClarityVS2+ColorI:ClarityVS2+ColorE:ClarityVVS1+ColorF:ClarityVVS1+ColorG:ClarityVVS1+ColorH:ClarityVVS1+ColorI:ClarityVVS1+ColorE:ClarityVVS2+ColorF:ClarityVVS2+ColorG:ClarityVVS2+ColorH:ClarityVVS2+ColorI:ClarityVVS2+ColorE:PolishG+ColorF:PolishG+ColorG:PolishG+ColorH:PolishG+ColorI:PolishG+ColorE:PolishID+ColorF:PolishID+ColorG:PolishID+ColorH:PolishID+ColorI:PolishID+ColorE:PolishVG+ColorF:PolishVG+ColorG:PolishVG+ColorH:PolishVG+ColorI:PolishVG+ColorE:SymmetryG+ColorF:SymmetryG+ColorG:SymmetryG+ColorH:SymmetryG+ColorI:SymmetryG+ColorE:SymmetryID+ColorF:SymmetryID+ColorG:SymmetryID+ColorH:SymmetryID+ColorI:SymmetryID+ColorE:SymmetryVG+ColorF:SymmetryVG+ColorG:SymmetryVG+ColorH:SymmetryVG+ColorI:SymmetryVG+ColorE:ReportGIA+ColorF:ReportGIA+ColorG:ReportGIA+ColorH:ReportGIA+ColorI:ReportGIA+PolishG:SymmetryG+PolishID:SymmetryG+PolishVG:SymmetryG+PolishG:SymmetryID+PolishID:SymmetryID+PolishVG:SymmetryID+PolishG:SymmetryVG+PolishID:SymmetryVG+PolishVG:SymmetryVG+PolishG:ReportGIA+PolishID:ReportGIA+PolishVG:ReportGIA+SymmetryG:ReportGIA+SymmetryID:ReportGIA+SymmetryVG:ReportGIA
          + LCarat:Cut + LCarat:Color + LCarat:Calarity + LCarat:Polish + LCarat:Symmetry + LCarat:Report
+ Caratbelow1:Cut + Caratbelow1:Color + Caratbelow1:Calarity + 
Caratbelow1:Polish + Caratbelow1:Symmetry + Caratbelow1:Report"


lm <- lm(formula, data = diamond.full.train)
summary(lm)
```

    ## 
    ## Call:
    ## lm(formula = formula, data = diamond.full.train)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -0.75294 -0.04332 -0.00037  0.04128  0.42819 
    ## 
    ## Coefficients: (25 not defined because of singularities)
    ##                                 Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                   10.8857669  0.0881621 123.474  < 2e-16 ***
    ## LCarat                         0.8411953  0.0434586  19.356  < 2e-16 ***
    ## recipCarat                    -1.0164058  0.0500397 -20.312  < 2e-16 ***
    ## Caratbelow1                   -0.3498842  0.0117198 -29.854  < 2e-16 ***
    ## Caratequal1                   -0.2629432  0.0113945 -23.076  < 2e-16 ***
    ## Caratbelow1.5                 -0.3022285  0.0099066 -30.508  < 2e-16 ***
    ## Caratequal1.5                 -0.2200981  0.0089504 -24.591  < 2e-16 ***
    ## Caratbelow2                   -0.2220632  0.0065940 -33.676  < 2e-16 ***
    ## Caratabove2                           NA         NA      NA       NA    
    ## CutFair                       -0.0356195  0.0443243  -0.804 0.421655    
    ## CutGood                       -0.0520513  0.0237103  -2.195 0.028181 *  
    ## CutIdeal                       0.0154857  0.0774769   0.200 0.841586    
    ## CutSignatureIdeal              0.2023201  0.0341848   5.918 3.43e-09 ***
    ## CutVeryGood                           NA         NA      NA       NA    
    ## ColorE                        -0.0325150  0.0530134  -0.613 0.539678    
    ## ColorF                        -0.1539324  0.0488881  -3.149 0.001648 ** 
    ## ColorG                        -0.3204833  0.0501157  -6.395 1.73e-10 ***
    ## ColorH                        -0.5015245  0.0491699 -10.200  < 2e-16 ***
    ## ColorI                        -0.5945498  0.0518050 -11.477  < 2e-16 ***
    ## ClarityIF                      0.0048389  0.0830966   0.058 0.953565    
    ## ClaritySI1                    -0.9776400  0.0616623 -15.855  < 2e-16 ***
    ## ClarityVS1                    -0.7176252  0.0634776 -11.305  < 2e-16 ***
    ## ClarityVS2                    -0.8327150  0.0621246 -13.404  < 2e-16 ***
    ## ClarityVVS1                   -0.3443668  0.0967819  -3.558 0.000376 ***
    ## ClarityVVS2                   -0.4466233  0.0549761  -8.124 5.46e-16 ***
    ## PolishG                       -0.0651099  0.0397003  -1.640 0.101052    
    ## PolishID                       0.1062700  0.0592420   1.794 0.072892 .  
    ## PolishVG                      -0.0350036  0.0254245  -1.377 0.168636    
    ## SymmetryG                     -0.0754902  0.0338489  -2.230 0.025771 *  
    ## SymmetryID                    -0.0115648  0.0592404  -0.195 0.845229    
    ## SymmetryVG                    -0.0545907  0.0298064  -1.832 0.067075 .  
    ## ReportGIA                      0.0783117  0.0463347   1.690 0.091056 .  
    ## CutGood:ColorE                 0.0027200  0.0272090   0.100 0.920375    
    ## CutIdeal:ColorE                0.0008022  0.0265122   0.030 0.975863    
    ## CutSignatureIdeal:ColorE      -0.0255896  0.0344844  -0.742 0.458079    
    ## CutVeryGood:ColorE             0.0049397  0.0258553   0.191 0.848492    
    ## CutGood:ColorF                -0.0183369  0.0285533  -0.642 0.520770    
    ## CutIdeal:ColorF               -0.0094570  0.0279715  -0.338 0.735303    
    ## CutSignatureIdeal:ColorF      -0.0471739  0.0348136  -1.355 0.175456    
    ## CutVeryGood:ColorF            -0.0081051  0.0272813  -0.297 0.766405    
    ## CutGood:ColorG                 0.0304481  0.0297336   1.024 0.305863    
    ## CutIdeal:ColorG                0.0131030  0.0291622   0.449 0.653220    
    ## CutSignatureIdeal:ColorG      -0.0174746  0.0350065  -0.499 0.617670    
    ## CutVeryGood:ColorG             0.0247805  0.0286015   0.866 0.386305    
    ## CutGood:ColorH                 0.0280249  0.0288146   0.973 0.330795    
    ## CutIdeal:ColorH                0.0172214  0.0281344   0.612 0.540487    
    ## CutSignatureIdeal:ColorH      -0.0090151  0.0348627  -0.259 0.795962    
    ## CutVeryGood:ColorH             0.0294948  0.0275418   1.071 0.284255    
    ## CutGood:ColorI                -0.0345030  0.0313305  -1.101 0.270830    
    ## CutIdeal:ColorI               -0.0658519  0.0308265  -2.136 0.032704 *  
    ## CutSignatureIdeal:ColorI      -0.0783534  0.0370452  -2.115 0.034466 *  
    ## CutVeryGood:ColorI            -0.0452575  0.0301711  -1.500 0.133661    
    ## CutGood:ClarityIF             -0.0221305  0.0652305  -0.339 0.734421    
    ## CutIdeal:ClarityIF             0.0020431  0.0966071   0.021 0.983128    
    ## CutSignatureIdeal:ClarityIF   -0.0898913  0.0687890  -1.307 0.191343    
    ## CutVeryGood:ClarityIF         -0.0457992  0.0623562  -0.734 0.462687    
    ## CutGood:ClaritySI1             0.0163125  0.0309851   0.526 0.598586    
    ## CutIdeal:ClaritySI1            0.0172088  0.0794706   0.217 0.828572    
    ## CutSignatureIdeal:ClaritySI1  -0.0016432  0.0324406  -0.051 0.959604    
    ## CutVeryGood:ClaritySI1        -0.0152513  0.0290058  -0.526 0.599046    
    ## CutGood:ClarityVS1             0.0631323  0.0348252   1.813 0.069909 .  
    ## CutIdeal:ClarityVS1            0.0910751  0.0808574   1.126 0.260056    
    ## CutSignatureIdeal:ClarityVS1   0.0389351  0.0352871   1.103 0.269906    
    ## CutVeryGood:ClarityVS1         0.0508347  0.0326680   1.556 0.119738    
    ## CutGood:ClarityVS2             0.0382766  0.0320417   1.195 0.232297    
    ## CutIdeal:ClarityVS2            0.0606787  0.0798011   0.760 0.447062    
    ## CutSignatureIdeal:ClarityVS2   0.0220058  0.0333829   0.659 0.509798    
    ## CutVeryGood:ClarityVS2         0.0201684  0.0299982   0.672 0.501408    
    ## CutGood:ClarityVVS1            0.0204895  0.0833308   0.246 0.805783    
    ## CutIdeal:ClarityVVS1           0.0813694  0.1081432   0.752 0.451827    
    ## CutSignatureIdeal:ClarityVVS1  0.0381010  0.0812481   0.469 0.639126    
    ## CutVeryGood:ClarityVVS1        0.0643305  0.0791771   0.812 0.416545    
    ## CutGood:ClarityVVS2                   NA         NA      NA       NA    
    ## CutIdeal:ClarityVVS2           0.0440963  0.0741990   0.594 0.552336    
    ## CutSignatureIdeal:ClarityVVS2         NA         NA      NA       NA    
    ## CutVeryGood:ClarityVVS2               NA         NA      NA       NA    
    ## CutGood:PolishG                0.0503255  0.0245587   2.049 0.040488 *  
    ## CutIdeal:PolishG               0.0094481  0.0262068   0.361 0.718472    
    ## CutSignatureIdeal:PolishG             NA         NA      NA       NA    
    ## CutVeryGood:PolishG            0.0229965  0.0237136   0.970 0.332205    
    ## CutGood:PolishID                      NA         NA      NA       NA    
    ## CutIdeal:PolishID              0.0589223  0.0628614   0.937 0.348624    
    ## CutSignatureIdeal:PolishID     0.0056629  0.0309285   0.183 0.854729    
    ## CutVeryGood:PolishID                  NA         NA      NA       NA    
    ## CutGood:PolishVG               0.0166423  0.0198450   0.839 0.401719    
    ## CutIdeal:PolishVG              0.0009330  0.0193670   0.048 0.961577    
    ## CutSignatureIdeal:PolishVG            NA         NA      NA       NA    
    ## CutVeryGood:PolishVG           0.0113083  0.0191188   0.591 0.554223    
    ## CutGood:SymmetryG              0.0037363  0.0269953   0.138 0.889926    
    ## CutIdeal:SymmetryG             0.0248965  0.0270551   0.920 0.357499    
    ## CutSignatureIdeal:SymmetryG           NA         NA      NA       NA    
    ## CutVeryGood:SymmetryG          0.0140799  0.0253589   0.555 0.578761    
    ## CutGood:SymmetryID            -0.0157786  0.0815211  -0.194 0.846534    
    ## CutIdeal:SymmetryID           -0.0651399  0.0648530  -1.004 0.315216    
    ## CutSignatureIdeal:SymmetryID          NA         NA      NA       NA    
    ## CutVeryGood:SymmetryID                NA         NA      NA       NA    
    ## CutGood:SymmetryVG             0.0005952  0.0267573   0.022 0.982255    
    ## CutIdeal:SymmetryVG            0.0078725  0.0252696   0.312 0.755402    
    ## CutSignatureIdeal:SymmetryVG          NA         NA      NA       NA    
    ## CutVeryGood:SymmetryVG         0.0179430  0.0252305   0.711 0.477012    
    ## CutGood:ReportGIA              0.0004958  0.0275888   0.018 0.985662    
    ## CutIdeal:ReportGIA             0.0035025  0.0311725   0.112 0.910544    
    ## CutSignatureIdeal:ReportGIA           NA         NA      NA       NA    
    ## CutVeryGood:ReportGIA          0.0040532  0.0258882   0.157 0.875593    
    ## ColorE:ClarityIF              -0.1694396  0.0235248  -7.203 6.65e-13 ***
    ## ColorF:ClarityIF              -0.2033339  0.0190663 -10.665  < 2e-16 ***
    ## ColorG:ClarityIF              -0.3079083  0.0166317 -18.513  < 2e-16 ***
    ## ColorH:ClarityIF              -0.3124721  0.0209664 -14.904  < 2e-16 ***
    ## ColorI:ClarityIF              -0.3500132  0.0223810 -15.639  < 2e-16 ***
    ## ColorE:ClaritySI1              0.0599987  0.0128979   4.652 3.36e-06 ***
    ## ColorF:ClaritySI1              0.0910804  0.0122141   7.457 1.01e-13 ***
    ## ColorG:ClaritySI1              0.1729081  0.0114902  15.048  < 2e-16 ***
    ## ColorH:ClaritySI1              0.3263847  0.0126096  25.884  < 2e-16 ***
    ## ColorI:ClaritySI1              0.3544162  0.0136537  25.958  < 2e-16 ***
    ## ColorE:ClarityVS1              0.0453761  0.0149639   3.032 0.002437 ** 
    ## ColorF:ClarityVS1              0.0789217  0.0138676   5.691 1.32e-08 ***
    ## ColorG:ClarityVS1              0.1208277  0.0126494   9.552  < 2e-16 ***
    ## ColorH:ClarityVS1              0.1622460  0.0140889  11.516  < 2e-16 ***
    ## ColorI:ClarityVS1              0.1520032  0.0150237  10.118  < 2e-16 ***
    ## ColorE:ClarityVS2              0.0819840  0.0138688   5.911 3.58e-09 ***
    ## ColorF:ClarityVS2              0.1231261  0.0129482   9.509  < 2e-16 ***
    ## ColorG:ClarityVS2              0.1954206  0.0119339  16.375  < 2e-16 ***
    ## ColorH:ClarityVS2              0.2490765  0.0133847  18.609  < 2e-16 ***
    ## ColorI:ClarityVS2              0.2416156  0.0142978  16.899  < 2e-16 ***
    ## ColorE:ClarityVVS1            -0.0128802  0.0223198  -0.577 0.563912    
    ## ColorF:ClarityVVS1            -0.0575130  0.0202213  -2.844 0.004468 ** 
    ## ColorG:ClarityVVS1            -0.0983140  0.0189195  -5.196 2.10e-07 ***
    ## ColorH:ClarityVVS1            -0.0931286  0.0211952  -4.394 1.13e-05 ***
    ## ColorI:ClarityVVS1            -0.0895725  0.0233886  -3.830 0.000130 ***
    ## ColorE:ClarityVVS2                    NA         NA      NA       NA    
    ## ColorF:ClarityVVS2                    NA         NA      NA       NA    
    ## ColorG:ClarityVVS2                    NA         NA      NA       NA    
    ## ColorH:ClarityVVS2                    NA         NA      NA       NA    
    ## ColorI:ClarityVVS2                    NA         NA      NA       NA    
    ## ColorE:PolishG                -0.0153979  0.0152779  -1.008 0.313565    
    ## ColorF:PolishG                -0.0252057  0.0145185  -1.736 0.082597 .  
    ## ColorG:PolishG                -0.0017981  0.0141086  -0.127 0.898592    
    ## ColorH:PolishG                 0.0053534  0.0149239   0.359 0.719821    
    ## ColorI:PolishG                 0.0069415  0.0150175   0.462 0.643936    
    ## ColorE:PolishID               -0.1821826  0.1087685  -1.675 0.093996 .  
    ## ColorF:PolishID               -0.0853109  0.0639314  -1.334 0.182121    
    ## ColorG:PolishID               -0.1044873  0.0638984  -1.635 0.102059    
    ## ColorH:PolishID               -0.0885673  0.0617982  -1.433 0.151863    
    ## ColorI:PolishID               -0.0900187  0.0692037  -1.301 0.193385    
    ## ColorE:PolishVG               -0.0009669  0.0095537  -0.101 0.919390    
    ## ColorF:PolishVG               -0.0132251  0.0090054  -1.469 0.142002    
    ## ColorG:PolishVG                0.0017243  0.0084322   0.204 0.837980    
    ## ColorH:PolishVG               -0.0027216  0.0089713  -0.303 0.761618    
    ## ColorI:PolishVG                0.0016703  0.0094480   0.177 0.859677    
    ## ColorE:SymmetryG              -0.0069342  0.0137792  -0.503 0.614817    
    ## ColorF:SymmetryG               0.0045517  0.0134013   0.340 0.734138    
    ## ColorG:SymmetryG               0.0031296  0.0127727   0.245 0.806449    
    ## ColorH:SymmetryG               0.0142903  0.0133548   1.070 0.284642    
    ## ColorI:SymmetryG               0.0086615  0.0140463   0.617 0.537497    
    ## ColorE:SymmetryID              0.0876309  0.1117377   0.784 0.432922    
    ## ColorF:SymmetryID              0.0525656  0.0642818   0.818 0.413541    
    ## ColorG:SymmetryID              0.0541963  0.0642913   0.843 0.399275    
    ## ColorH:SymmetryID              0.0157902  0.0636482   0.248 0.804076    
    ## ColorI:SymmetryID              0.0440599  0.0698823   0.630 0.528400    
    ## ColorE:SymmetryVG             -0.0013358  0.0100982  -0.132 0.894768    
    ## ColorF:SymmetryVG              0.0137304  0.0095470   1.438 0.150433    
    ## ColorG:SymmetryVG              0.0079407  0.0089178   0.890 0.373271    
    ## ColorH:SymmetryVG              0.0183679  0.0094493   1.944 0.051962 .  
    ## ColorI:SymmetryVG              0.0081760  0.0100294   0.815 0.414986    
    ## ColorE:ReportGIA              -0.0805410  0.0480703  -1.675 0.093893 .  
    ## ColorF:ReportGIA              -0.0170635  0.0428946  -0.398 0.690792    
    ## ColorG:ReportGIA              -0.0306871  0.0430828  -0.712 0.476318    
    ## ColorH:ReportGIA              -0.0686785  0.0436122  -1.575 0.115368    
    ## ColorI:ReportGIA              -0.0461353  0.0427344  -1.080 0.280373    
    ## PolishG:SymmetryG              0.0336732  0.0162825   2.068 0.038678 *  
    ## PolishID:SymmetryG                    NA         NA      NA       NA    
    ## PolishVG:SymmetryG             0.0194432  0.0096454   2.016 0.043867 *  
    ## PolishG:SymmetryID                    NA         NA      NA       NA    
    ## PolishID:SymmetryID           -0.0205458  0.0386648  -0.531 0.595173    
    ## PolishVG:SymmetryID                   NA         NA      NA       NA    
    ## PolishG:SymmetryVG             0.0275811  0.0145577   1.895 0.058194 .  
    ## PolishID:SymmetryVG                   NA         NA      NA       NA    
    ## PolishVG:SymmetryVG            0.0255225  0.0053690   4.754 2.05e-06 ***
    ## PolishG:ReportGIA             -0.0091702  0.0286668  -0.320 0.749065    
    ## PolishID:ReportGIA                    NA         NA      NA       NA    
    ## PolishVG:ReportGIA            -0.0056570  0.0175056  -0.323 0.746592    
    ## SymmetryG:ReportGIA            0.0190722  0.0224795   0.848 0.396234    
    ## SymmetryID:ReportGIA                  NA         NA      NA       NA    
    ## SymmetryVG:ReportGIA           0.0018631  0.0177234   0.105 0.916282    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.07102 on 5841 degrees of freedom
    ## Multiple R-squared:  0.9903, Adjusted R-squared:   0.99 
    ## F-statistic:  3762 on 158 and 5841 DF,  p-value: < 2.2e-16

``` r
lm.pred <- predict(lm, diamond.full.test)
accuracy(exp(lm.pred), diamond.full.test$Price)
```

    ##                ME     RMSE      MAE         MPE     MAPE
    ## Test set 95.15284 1231.742 668.8626 -0.03850604 5.348386

``` r
lm.step <- step(lm, direction = "backward", trace=0, step=10)
lm.step.pred <- predict(lm.step, diamond.full.test)
accuracy(exp(lm.step.pred), diamond.full.test$Price)
```

    ##                ME     RMSE      MAE         MPE     MAPE
    ## Test set 95.15284 1231.742 668.8626 -0.03850604 5.348386

Tree-based models
-----------------

Here we utilize several tree-based model to predict log price of the diamonds based on several characteristics present in the data.

### Single Tuned Tree

Our initial tuned tree (best cp is around 0.00000151859) yields a MAPE of 7.0% when applied to the validation set. The importance variable list shows that log carat size, inverse of carat size, as well as the bins of carat size are all quite significant variables in predicting price.

Note that throughout our modeling analysis we will utilize a common model specification that each process with start with when fitting.

``` r
model_formula <- "LPrice ~ LCarat +  recipCarat + Cut + Color + Clarity + Polish + Symmetry + 
                           Report + Caratbelow1 + Caratequal1 + Caratbelow1.5 +
                           Caratequal1.5 + Caratbelow2 + Caratabove2"
```

``` r
rt.auto.cv <- rpart(model_formula, data = diamond.train, 
                    control = rpart.control(cp = 0.000001, xval = 10))  
#xval is number of folds in the K-fold cross-validation.
#printcp(rt.auto.cv)  # Print out the cp table of cross-validation errors.

#The R-squared for a regression tree is 1 minus rel error. 
#xerror (or relative cross-validation error where "x" stands for "cross") is a scaled 
#version of overall average of the 5 out-of-sample MSEs across the 5 folds. 
#For the scaling, the MSE's are divided by the "root node error" of 0.091868, 
#which is the variance in the y's. 
#xstd measures the variation in xerror between the folds. nsplit is the number of terminal nodes minus 1.

plotcp(rt.auto.cv)  
```

![](diamond-analysis_files/figure-markdown_github/autofitting-rpart-tree-1.png)

``` r
# The horizontal line in this plot is one standard deviation above 
# the minimum xerror value in the cp table. Because simpler trees are better, 
# the convention is to choose the cp level to the left of the cp level with the 
# minimum xerror that is first above the line. 

# In this case, the minimum xerror is 0.3972833 at row 35 in the cp table.
rt.auto.cv.table <- as.data.frame(rt.auto.cv$cptable)
min(rt.auto.cv.table$xerror)
```

    ## [1] 0.01678595

``` r
bestcp <- rt.auto.cv.table$CP[rt.auto.cv.table$xerror==min(rt.auto.cv.table$xerror)]

# According to this analysis using 5-fold cross-validation, setting cp = 0.002869198 is best. 
# Take a look at the resulting 18-terminal-node tree.
rt.tuned.opt.cv <- rpart(model_formula, data = diamond.train, 
                         control = rpart.control(cp = bestcp))
prp(rt.tuned.opt.cv, type = 1, extra = 1)
```

![](diamond-analysis_files/figure-markdown_github/autofitting-rpart-tree-2.png)

``` r
importance <- as.data.frame(rt.tuned.opt.cv$variable.importance)
importance
```

    ##               rt.tuned.opt.cv$variable.importance
    ## LCarat                                2626.734670
    ## recipCarat                            2626.527258
    ## Caratabove2                           1455.002927
    ## Caratbelow2                            777.979267
    ## Caratbelow1.5                          748.017341
    ## Caratbelow1                            291.915230
    ## Clarity                                220.101351
    ## Caratequal1.5                          182.817922
    ## Color                                  158.329191
    ## Cut                                     24.261775
    ## Symmetry                                23.435850
    ## Polish                                  19.696325
    ## Report                                   7.323699
    ## Caratequal1                              1.373455

``` r
rt.tuned.opt.cv.pred <- predict(rt.tuned.opt.cv, diamond.test)
accuracy(exp(rt.tuned.opt.cv.pred), diamond.test$Price)
```

    ##               ME    RMSE     MAE       MPE    MAPE
    ## Test set 21.0808 1666.28 852.826 -0.124237 6.44278

To facilitate some intuition of the variables, here we generate a few simpler trees than the model above. These trees have much larger cp parameters and as such have much fewer layers, which aids with interpretability.

``` r
# fitting four simple trees using different complexity parameters
rt.simple.tree1 <- rpart(model_formula, data = diamond.train, 
                         control = rpart.control(cp = 0.005))
rt.simple.tree2 <- rpart(model_formula, data = diamond.train, 
                         control = rpart.control(cp = 0.001))
rt.simple.tree3 <- rpart(model_formula, data = diamond.train, 
                         control = rpart.control(cp = 0.0005))
rt.simple.tree4 <- rpart(model_formula, data = diamond.train, 
                         control = rpart.control(cp = 0.0001))
```

Plots of the trees and diagnostics are available in the `output` folder of this analysis.

### Bagged Tree/Random Forest

The second tree-based method is a bagged tree, which we implement with the `randomForest()` function and the `mtry` argument set equal to 14 - the number of explanatory variables feed into the model.

``` r
#bag with smaller train dataset#
bag.tree <- randomForest(as.formula(model_formula), 
                         data=diamond.smaller.train, mtry=14, ntree=100,
                         importance=TRUE)
bag.tree.pred.valid <- predict(bag.tree, newdata=diamond.validation)
accuracy(exp(bag.tree.pred.valid), diamond.validation$Price)
```

    ##                ME    RMSE      MAE        MPE     MAPE
    ## Test set 54.93479 1145.96 651.0374 -0.3755985 5.467027

This bagged tree yields a MAPE of 5.34% on the validation set, already a great improvement from the 6.4% of the single tuned tree. Given the improvement of the bagged tree, we could estimate the bagged tree on the full training set by feeding that dataset to the `randomForest()` function like so:

``` r
bag.tree <- randomForest(as.formula(model_formula), 
                         data=diamond.train, mtry=14, ntree=100,
                         importance=TRUE)
bag.tree.pred <- predict(bag.tree, newdata=diamond.test)
accuracy(exp(bag.tree.pred), diamond.test$Price)
```

### Random Forest

The third tree-based model we implement is a cross validated random forest, which decorrelates the tree and should provide additional improvements over the bagged tree method.

``` r
# k-folds cross validation automatically using rfcv
trainx <- diamond.smaller.train[,c("LCarat", "recipCarat", "Cut", "Color", "Clarity", "Polish", "Symmetry",
                                  "Report", "Caratbelow1", "Caratequal1", "Caratbelow1.5","Caratequal1.5", 
                                  "Caratbelow2", "Caratabove2")]
trainy <- diamond.smaller.train$LPrice
random.forest.cv <- rfcv(trainx, trainy,
                         cv.folds = 10, scale="unit", step=-1, ntree=100)
plot(x=1:14, y=rev(random.forest.cv$error.cv),
     xlab="mtry parameter", ylab="Cross Validation Error",
     main="Random Forest Cross Validation Results")
```

![](diamond-analysis_files/figure-markdown_github/prepare-data-for-rf-cv-training-1.png)

The cross validation results above shows that the best number of `mtry` for random forest should be 9 (vs. 14). We will use this value when estimating our random forest model.

``` r
random.forest.cv$error.cv[random.forest.cv$error.cv==min(random.forest.cv$error.cv)]
```

    ##         9 
    ## 0.0103964

``` r
random.forest.cv.1 <- randomForest(as.formula(model_formula), 
                                   data=diamond.smaller.train, mtry=9, ntree=100,
                                   importance=TRUE)
random.forest.cv.1.pred.valid <- predict(random.forest.cv.1, newdata=diamond.validation)
accuracy(exp(random.forest.cv.1.pred.valid), diamond.validation$Price)
```

    ##               ME    RMSE    MAE       MPE    MAPE
    ## Test set 76.0959 1149.84 648.61 -0.398527 5.40707

Finally, we can repeat the same procedures above on the full training set.

``` r
# perform cross validation to tune the model parameters
trainx <- diamond.train[,c("LCarat", "recipCarat", "Cut", "Color", "Clarity", "Polish", "Symmetry",
                           "Report", "Caratbelow1", "Caratequal1", "Caratbelow1.5","Caratequal1.5", 
                           "Caratbelow2", "Caratabove2")]
trainy <- diamond.train$LPrice
random.forest.cv <- rfcv(trainx, 
                         trainy,
                         cv.folds=10, scale="unit", step=-1, ntree=100)

# determine the best fitting model
random.forest.cv$error.cv
length(random.forest.cv$error.cv)
plot(x=1:14, y=rev(random.forest.cv$error.cv),
     xlab="mtry parameter", ylab="Cross Validation Error",
     main="Random Forest Cross Validation Results")
random.forest.cv$error.cv[random.forest.cv$error.cv==min(random.forest.cv$error.cv)]

# use the optimal parameters to fit the final model
random.forest.cv.1 <- randomForest(as.formula(model_formula), 
                                   data=diamond.train, mtry=9, ntree=100,
                                   importance=TRUE)
# measure the accuracy
random.forest.cv.1.pred <- predict(random.forest.cv.1, newdata=diamond.test)
accuracy(exp(random.forest.cv.1.pred), diamond.test$Price)
```

### Boosted Trees

The last tree-based model we will be using is a boosted tree model. We use cross validation to identify the best value for the parameter `n.trees`, which turns out to be 5,207.

``` r
boost <- gbm(as.formula(model_formula), data=diamond.smaller.train,
             distribution = "gaussian",
             n.trees=100, interaction.depth=6, cv.folds=10, shrinkage = 0.011)
plot(boost$cv.error)
```

![](diamond-analysis_files/figure-markdown_github/boosted-tree%20cv-1.png)

``` r
best_iteration <- which(boost$cv.error==min(boost$cv.error))
best_iteration <- 5207
```

Using this `n.trees` parameter, we estimate the model on the smaller training set using 100 iterations, which yields a MAPE of 4.46% on the validation set, representing additional improvements over the random forest model. It looks like the model is continually getting better even at the 100th iteration. More iterations might help us find the true optimum number of trees to minimize prediction error.

``` r
boost.cv <- gbm(as.formula(model_formula), data=diamond.smaller.train,
                distribution = "gaussian",
                n.trees=best_iteration, interaction.depth=6, cv.folds=10, shrinkage = 0.011)
boost.cv.pred.valid <- predict(boost.cv, newdata=diamond.validation, n.trees=best_iteration)
accuracy(exp(boost.cv.pred.valid), diamond.validation$Price)
```

    ##                ME     RMSE      MAE        MPE    MAPE
    ## Test set 28.32441 971.3675 539.8008 -0.2647916 4.42206

Finally, we repeat the same procedures above using the full dataset, including cross validation. Cross validation shows that 100 is the best value for `n.trees`, and using this parameter yields a MAPE of 4.23808% on the test set.

``` r
boost <- gbm(as.formula(model_formula), data=diamond.train,
             distribution = "gaussian",
             n.trees=100, interaction.depth=6, cv.folds=10, shrinkage = 0.011)
best_iteration <- which(boost$cv.error==min(boost$cv.error))
best_iteration <- 5207
boost.cv <- gbm(as.formula(model_formula), data=diamond.train,
                distribution = "gaussian",
                n.trees=best_iteration, interaction.depth=6, cv.folds=10, shrinkage = 0.011)
boost.cv.pred <- predict(boost.cv, newdata=diamond.test, n.trees=best_iteration)
accuracy(exp(boost.cv.pred), diamond.test$Price)
```

### Lasso Regression

Another method we could use to choose which variable to include is the Lasso regression. Here we use cross-validation to determine the best lambda parameter used in Lasso regression to "regularize" the coefficients of the variables included.

``` r
#smaller train dataset
xtrain <- as.matrix(diamond.full.smaller.train[, -c(1:11)])
ytrain <- as.vector(diamond.full.smaller.train$LPrice)
xtest <- as.matrix(diamond.full.validation[, -c(1:11)])
lm.regularized.cv <- cv.glmnet(xtrain, ytrain, 
                               nfolds = 10, family = "gaussian", alpha=1)
```

``` r
lm.regularized.cv$lambda.min
```

    ## [1] 0.0002032383

``` r
(minLogLambda <- log(lm.regularized.cv$lambda.min))
```

    ## [1] -8.501132

``` r
coef(lm.regularized.cv, s = "lambda.min")  
```

    ## 184 x 1 sparse Matrix of class "dgCMatrix"
    ##                                           1
    ## (Intercept)                   10.1296499779
    ## LCarat                         0.9214650900
    ## recipCarat                    -0.9397805888
    ## Caratbelow1                   -0.0890530060
    ## Caratequal1                    .           
    ## Caratbelow1.5                 -0.0415595887
    ## Caratequal1.5                  0.0289034530
    ## Caratbelow2                    0.0321461729
    ## Caratabove2                    0.2424561741
    ## CutFair                       -0.0805880114
    ## CutGood                       -0.0411617051
    ## CutIdeal                       0.0487787086
    ## CutSignatureIdeal              0.2412910556
    ## CutVeryGood                    .           
    ## ColorE                        -0.0383020520
    ## ColorF                        -0.0936333664
    ## ColorG                        -0.1544565355
    ## ColorH                        -0.2814109171
    ## ColorI                        -0.4335001162
    ## ClarityIF                      0.3908236345
    ## ClaritySI1                    -0.5154675428
    ## ClarityVS1                    -0.2149127091
    ## ClarityVS2                    -0.3490084217
    ## ClarityVVS1                    0.1188945345
    ## ClarityVVS2                    .           
    ## PolishG                       -0.0206835108
    ## PolishID                       .           
    ## PolishVG                      -0.0216565437
    ## SymmetryG                     -0.0188618494
    ## SymmetryID                     0.0072880898
    ## SymmetryVG                    -0.0226769724
    ## ReportGIA                      0.0444939536
    ## CutGood:ColorE                -0.0003584597
    ## CutIdeal:ColorE               -0.0051809477
    ## CutSignatureIdeal:ColorE      -0.0593350453
    ## CutVeryGood:ColorE             .           
    ## CutGood:ColorF                -0.0087265497
    ## CutIdeal:ColorF               -0.0019737895
    ## CutSignatureIdeal:ColorF      -0.0664279125
    ## CutVeryGood:ColorF             .           
    ## CutGood:ColorG                 0.0098333287
    ## CutIdeal:ColorG               -0.0068115325
    ## CutSignatureIdeal:ColorG      -0.0654090260
    ## CutVeryGood:ColorG             .           
    ## CutGood:ColorH                 0.0082834413
    ## CutIdeal:ColorH                .           
    ## CutSignatureIdeal:ColorH      -0.0447474595
    ## CutVeryGood:ColorH             0.0049675116
    ## CutGood:ColorI                 .           
    ## CutIdeal:ColorI               -0.0300909237
    ## CutSignatureIdeal:ColorI      -0.0742447057
    ## CutVeryGood:ColorI            -0.0128689827
    ## CutGood:ClarityIF              0.0213824589
    ## CutIdeal:ClarityIF             0.0105348559
    ## CutSignatureIdeal:ClarityIF   -0.0498189869
    ## CutVeryGood:ClarityIF          .           
    ## CutGood:ClaritySI1            -0.0012088780
    ## CutIdeal:ClaritySI1           -0.0233626424
    ## CutSignatureIdeal:ClaritySI1  -0.0291583877
    ## CutVeryGood:ClaritySI1        -0.0196997949
    ## CutGood:ClarityVS1            -0.0029688404
    ## CutIdeal:ClarityVS1            .           
    ## CutSignatureIdeal:ClarityVS1  -0.0248875257
    ## CutVeryGood:ClarityVS1         .           
    ## CutGood:ClarityVS2             .           
    ## CutIdeal:ClarityVS2           -0.0008169613
    ## CutSignatureIdeal:ClarityVS2  -0.0150840547
    ## CutVeryGood:ClarityVS2         .           
    ## CutGood:ClarityVVS1            .           
    ## CutIdeal:ClarityVVS1           0.0079828378
    ## CutSignatureIdeal:ClarityVVS1 -0.0036761998
    ## CutVeryGood:ClarityVVS1        0.0268077675
    ## CutGood:ClarityVVS2           -0.0142219946
    ## CutIdeal:ClarityVVS2           0.0006979062
    ## CutSignatureIdeal:ClarityVVS2 -0.0075528107
    ## CutVeryGood:ClarityVVS2       -0.0006009588
    ## CutGood:PolishG                0.0130646016
    ## CutIdeal:PolishG              -0.0298790805
    ## CutSignatureIdeal:PolishG      .           
    ## CutVeryGood:PolishG           -0.0090119910
    ## CutGood:PolishID               .           
    ## CutIdeal:PolishID              .           
    ## CutSignatureIdeal:PolishID     0.0033665491
    ## CutVeryGood:PolishID           .           
    ## CutGood:PolishVG               0.0007961749
    ## CutIdeal:PolishVG             -0.0113223588
    ## CutSignatureIdeal:PolishVG     .           
    ## CutVeryGood:PolishVG          -0.0020541060
    ## CutGood:SymmetryG             -0.0098122013
    ## CutIdeal:SymmetryG             0.0092758234
    ## CutSignatureIdeal:SymmetryG    .           
    ## CutVeryGood:SymmetryG          .           
    ## CutGood:SymmetryID             .           
    ## CutIdeal:SymmetryID            .           
    ## CutSignatureIdeal:SymmetryID   .           
    ## CutVeryGood:SymmetryID         .           
    ## CutGood:SymmetryVG            -0.0202833631
    ## CutIdeal:SymmetryVG           -0.0139026694
    ## CutSignatureIdeal:SymmetryVG   .           
    ## CutVeryGood:SymmetryVG         .           
    ## CutGood:ReportGIA              .           
    ## CutIdeal:ReportGIA             0.0025624743
    ## CutSignatureIdeal:ReportGIA    0.0065179955
    ## CutVeryGood:ReportGIA          .           
    ## ColorE:ClarityIF              -0.1922409640
    ## ColorF:ClarityIF              -0.2542910002
    ## ColorG:ClarityIF              -0.4621961127
    ## ColorH:ClarityIF              -0.5231291060
    ## ColorI:ClarityIF              -0.5581577511
    ## ColorE:ClaritySI1              .           
    ## ColorF:ClaritySI1              .           
    ## ColorG:ClaritySI1              .           
    ## ColorH:ClaritySI1              0.0766830832
    ## ColorI:ClaritySI1              0.1130198413
    ## ColorE:ClarityVS1             -0.0086781518
    ## ColorF:ClarityVS1             -0.0015706919
    ## ColorG:ClarityVS1             -0.0481612215
    ## ColorH:ClarityVS1             -0.0772251437
    ## ColorI:ClarityVS1             -0.0787575580
    ## ColorE:ClarityVS2              0.0130823884
    ## ColorF:ClarityVS2              0.0330873826
    ## ColorG:ClarityVS2              0.0089180409
    ## ColorH:ClarityVS2              .           
    ## ColorI:ClarityVS2              .           
    ## ColorE:ClarityVVS1            -0.0338626704
    ## ColorF:ClarityVVS1            -0.1007152364
    ## ColorG:ClarityVVS1            -0.2399942331
    ## ColorH:ClarityVVS1            -0.3133946772
    ## ColorI:ClarityVVS1            -0.2997482990
    ## ColorE:ClarityVVS2            -0.0416028567
    ## ColorF:ClarityVVS2            -0.0757254689
    ## ColorG:ClarityVVS2            -0.1586946148
    ## ColorH:ClarityVVS2            -0.2331569338
    ## ColorI:ClarityVVS2            -0.2250931887
    ## ColorE:PolishG                -0.0121515196
    ## ColorF:PolishG                -0.0160507805
    ## ColorG:PolishG                 .           
    ## ColorH:PolishG                 0.0087705706
    ## ColorI:PolishG                 0.0035895055
    ## ColorE:PolishID               -0.0019605561
    ## ColorF:PolishID                .           
    ## ColorG:PolishID                0.0015667862
    ## ColorH:PolishID               -0.0124884948
    ## ColorI:PolishID                0.0104582771
    ## ColorE:PolishVG                .           
    ## ColorF:PolishVG               -0.0095774807
    ## ColorG:PolishVG                0.0007964316
    ## ColorH:PolishVG               -0.0027502053
    ## ColorI:PolishVG                0.0030273334
    ## ColorE:SymmetryG              -0.0146892070
    ## ColorF:SymmetryG              -0.0097497317
    ## ColorG:SymmetryG              -0.0083133881
    ## ColorH:SymmetryG               .           
    ## ColorI:SymmetryG               .           
    ## ColorE:SymmetryID              .           
    ## ColorF:SymmetryID              .           
    ## ColorG:SymmetryID              .           
    ## ColorH:SymmetryID              .           
    ## ColorI:SymmetryID              0.0020176332
    ## ColorE:SymmetryVG             -0.0028950534
    ## ColorF:SymmetryVG              0.0045055483
    ## ColorG:SymmetryVG              0.0009504404
    ## ColorH:SymmetryVG              0.0143215935
    ## ColorI:SymmetryVG              0.0002928970
    ## ColorE:ReportGIA              -0.0167499666
    ## ColorF:ReportGIA               .           
    ## ColorG:ReportGIA               .           
    ## ColorH:ReportGIA              -0.0242910660
    ## ColorI:ReportGIA              -0.0044417898
    ## PolishG:SymmetryG              .           
    ## PolishID:SymmetryG             .           
    ## PolishVG:SymmetryG             0.0025283720
    ## PolishG:SymmetryID             .           
    ## PolishID:SymmetryID            0.0070312401
    ## PolishVG:SymmetryID            .           
    ## PolishG:SymmetryVG             0.0018016725
    ## PolishID:SymmetryVG            .           
    ## PolishVG:SymmetryVG            0.0164193946
    ## PolishG:ReportGIA              .           
    ## PolishID:ReportGIA             .           
    ## PolishVG:ReportGIA             .           
    ## SymmetryG:ReportGIA            .           
    ## SymmetryID:ReportGIA           .           
    ## SymmetryVG:ReportGIA           .

``` r
plot(lm.regularized.cv, label = TRUE)
abline(v = minLogLambda)
```

![](diamond-analysis_files/figure-markdown_github/unnamed-chunk-3-1.png)

``` r
lm.regularized <- glmnet(xtrain, ytrain, family = "gaussian", 
                         lambda=lm.regularized.cv$lambda.min)
plot(lm.regularized, xvar = "lambda", label = TRUE)
```

![](diamond-analysis_files/figure-markdown_github/unnamed-chunk-4-1.png)

``` r
lm.regularized.cv.pred.valid <- predict(lm.regularized.cv, newx = xtest, s = "lambda.min") 
lm.regularized.pred.valid <- predict(lm.regularized, newx = xtest, s = "lambda.min") 
```

``` r
accuracy(exp(as.ts(lm.regularized.cv.pred.valid)), as.ts(diamond.full.validation$Price))
```

    ##                ME     RMSE      MAE       MPE     MAPE       ACF1
    ## Test set 37.53947 1142.082 659.9324 -0.595341 5.503973 0.02535147
    ##           Theil's U
    ## Test set 0.09129714

``` r
accuracy(exp(as.ts(lm.regularized.pred.valid)), as.ts(diamond.full.validation$Price))
```

    ##                ME     RMSE      MAE        MPE     MAPE       ACF1
    ## Test set 38.65402 1136.629 657.0175 -0.5803963 5.493791 0.02744016
    ##           Theil's U
    ## Test set 0.09111464

``` r
#full dataset
xtrain <- as.matrix(diamond.full.train[, -c(1:11)])
ytrain <- as.vector(diamond.full.train$LPrice)
xtest <- as.matrix(diamond.full.test[, -c(1:11)])
lm.regularized.cv <- cv.glmnet(xtrain, ytrain, 
                               nfolds = 10, family = "gaussian", alpha=1)  # Fits the Lasso.
```

``` r
lm.regularized.cv$lambda.min
```

    ## [1] 0.0002021391

``` r
(minLogLambda <- log(lm.regularized.cv$lambda.min))
```

    ## [1] -8.506555

``` r
coef(lm.regularized.cv, s = "lambda.min")  
```

    ## 184 x 1 sparse Matrix of class "dgCMatrix"
    ##                                           1
    ## (Intercept)                    1.006621e+01
    ## LCarat                         9.148887e-01
    ## recipCarat                    -9.436316e-01
    ## Caratbelow1                   -8.813393e-02
    ## Caratequal1                    .           
    ## Caratbelow1.5                 -3.943500e-02
    ## Caratequal1.5                  3.692689e-02
    ## Caratbelow2                    3.493642e-02
    ## Caratabove2                    2.475245e-01
    ## CutFair                       -8.560336e-02
    ## CutGood                       -3.866033e-02
    ## CutIdeal                       5.486208e-02
    ## CutSignatureIdeal              2.628659e-01
    ## CutVeryGood                    .           
    ## ColorE                        -2.312634e-02
    ## ColorF                        -8.855708e-02
    ## ColorG                        -1.543037e-01
    ## ColorH                        -2.791086e-01
    ## ColorI                        -4.241403e-01
    ## ClarityIF                      4.613833e-01
    ## ClaritySI1                    -4.458845e-01
    ## ClarityVS1                    -1.548131e-01
    ## ClarityVS2                    -2.883360e-01
    ## ClarityVVS1                    1.632152e-01
    ## ClarityVVS2                    5.438780e-02
    ## PolishG                       -2.655940e-02
    ## PolishID                       7.377213e-03
    ## PolishVG                      -2.385848e-02
    ## SymmetryG                     -2.743692e-02
    ## SymmetryID                     9.740451e-03
    ## SymmetryVG                    -2.416680e-02
    ## ReportGIA                      4.374014e-02
    ## CutGood:ColorE                -2.519717e-03
    ## CutIdeal:ColorE               -3.709731e-03
    ## CutSignatureIdeal:ColorE      -5.257880e-02
    ## CutVeryGood:ColorE             .           
    ## CutGood:ColorF                -1.046267e-02
    ## CutIdeal:ColorF               -1.791566e-03
    ## CutSignatureIdeal:ColorF      -6.271931e-02
    ## CutVeryGood:ColorF             .           
    ## CutGood:ColorG                 3.591199e-03
    ## CutIdeal:ColorG               -1.237647e-02
    ## CutSignatureIdeal:ColorG      -6.328838e-02
    ## CutVeryGood:ColorG             .           
    ## CutGood:ColorH                 7.513970e-04
    ## CutIdeal:ColorH               -1.140406e-02
    ## CutSignatureIdeal:ColorH      -5.789720e-02
    ## CutVeryGood:ColorH             2.806551e-03
    ## CutGood:ColorI                 .           
    ## CutIdeal:ColorI               -3.249422e-02
    ## CutSignatureIdeal:ColorI      -6.373462e-02
    ## CutVeryGood:ColorI            -1.024242e-02
    ## CutGood:ClarityIF              1.947321e-03
    ## CutIdeal:ClarityIF             5.387566e-03
    ## CutSignatureIdeal:ClarityIF   -7.251095e-02
    ## CutVeryGood:ClarityIF          .           
    ## CutGood:ClaritySI1            -1.157560e-02
    ## CutIdeal:ClaritySI1           -3.440465e-02
    ## CutSignatureIdeal:ClaritySI1  -4.667193e-02
    ## CutVeryGood:ClaritySI1        -2.630818e-02
    ## CutGood:ClarityVS1            -3.805711e-03
    ## CutIdeal:ClarityVS1            .           
    ## CutSignatureIdeal:ClarityVS1  -4.472831e-02
    ## CutVeryGood:ClarityVS1         .           
    ## CutGood:ClarityVS2             .           
    ## CutIdeal:ClarityVS2           -4.580085e-04
    ## CutSignatureIdeal:ClarityVS2  -3.339764e-02
    ## CutVeryGood:ClarityVS2         .           
    ## CutGood:ClarityVVS1           -1.452653e-02
    ## CutIdeal:ClarityVVS1           1.704999e-02
    ## CutSignatureIdeal:ClarityVVS1 -1.503007e-02
    ## CutVeryGood:ClarityVVS1        3.968879e-02
    ## CutGood:ClarityVVS2           -1.405061e-02
    ## CutIdeal:ClarityVVS2           4.069045e-03
    ## CutSignatureIdeal:ClarityVVS2 -3.038980e-02
    ## CutVeryGood:ClarityVVS2        .           
    ## CutGood:PolishG                2.124424e-02
    ## CutIdeal:PolishG              -1.166110e-02
    ## CutSignatureIdeal:PolishG      .           
    ## CutVeryGood:PolishG            .           
    ## CutGood:PolishID               .           
    ## CutIdeal:PolishID              .           
    ## CutSignatureIdeal:PolishID     1.487523e-03
    ## CutVeryGood:PolishID           .           
    ## CutGood:PolishVG               .           
    ## CutIdeal:PolishVG             -1.228520e-02
    ## CutSignatureIdeal:PolishVG     .           
    ## CutVeryGood:PolishVG           .           
    ## CutGood:SymmetryG             -3.390055e-03
    ## CutIdeal:SymmetryG             6.012095e-03
    ## CutSignatureIdeal:SymmetryG    .           
    ## CutVeryGood:SymmetryG          .           
    ## CutGood:SymmetryID             .           
    ## CutIdeal:SymmetryID            .           
    ## CutSignatureIdeal:SymmetryID   .           
    ## CutVeryGood:SymmetryID         2.783513e-03
    ## CutGood:SymmetryVG            -1.179723e-02
    ## CutIdeal:SymmetryVG           -9.840608e-03
    ## CutSignatureIdeal:SymmetryVG   .           
    ## CutVeryGood:SymmetryVG         .           
    ## CutGood:ReportGIA              .           
    ## CutIdeal:ReportGIA             2.893596e-03
    ## CutSignatureIdeal:ReportGIA    .           
    ## CutVeryGood:ReportGIA          .           
    ## ColorE:ClarityIF              -2.098818e-01
    ## ColorF:ClarityIF              -2.733450e-01
    ## ColorG:ClarityIF              -4.585958e-01
    ## ColorH:ClarityIF              -5.362152e-01
    ## ColorI:ClarityIF              -5.691202e-01
    ## ColorE:ClaritySI1              .           
    ## ColorF:ClaritySI1              .           
    ## ColorG:ClaritySI1              .           
    ## ColorH:ClaritySI1              7.957892e-02
    ## ColorI:ClaritySI1              1.139905e-01
    ## ColorE:ClarityVS1             -6.739577e-03
    ## ColorF:ClarityVS1             -4.398147e-03
    ## ColorG:ClarityVS1             -4.366934e-02
    ## ColorH:ClarityVS1             -7.404192e-02
    ## ColorI:ClarityVS1             -7.889427e-02
    ## ColorE:ClarityVS2              1.649734e-02
    ## ColorF:ClarityVS2              2.742210e-02
    ## ColorG:ClarityVS2              1.776453e-02
    ## ColorH:ClarityVS2              .           
    ## ColorI:ClarityVS2              .           
    ## ColorE:ClarityVVS1            -3.589999e-02
    ## ColorF:ClarityVVS1            -1.111632e-01
    ## ColorG:ClarityVVS1            -2.331935e-01
    ## ColorH:ClarityVVS1            -3.016618e-01
    ## ColorI:ClarityVVS1            -2.902913e-01
    ## ColorE:ClarityVVS2            -4.117982e-02
    ## ColorF:ClarityVVS2            -7.138984e-02
    ## ColorG:ClarityVVS2            -1.535714e-01
    ## ColorH:ClarityVVS2            -2.259231e-01
    ## ColorI:ClarityVVS2            -2.178152e-01
    ## ColorE:PolishG                -1.422146e-02
    ## ColorF:PolishG                -2.130144e-02
    ## ColorG:PolishG                 .           
    ## ColorH:PolishG                 3.675351e-03
    ## ColorI:PolishG                 3.619510e-03
    ## ColorE:PolishID               -2.076484e-02
    ## ColorF:PolishID                .           
    ## ColorG:PolishID               -2.098128e-03
    ## ColorH:PolishID               -4.901632e-03
    ## ColorI:PolishID                .           
    ## ColorE:PolishVG                .           
    ## ColorF:PolishVG               -1.021185e-02
    ## ColorG:PolishVG                9.592824e-04
    ## ColorH:PolishVG               -4.254773e-05
    ## ColorI:PolishVG                .           
    ## ColorE:SymmetryG              -8.592366e-03
    ## ColorF:SymmetryG              -2.731812e-03
    ## ColorG:SymmetryG              -1.960517e-03
    ## ColorH:SymmetryG               4.199561e-03
    ## ColorI:SymmetryG               3.741494e-03
    ## ColorE:SymmetryID              .           
    ## ColorF:SymmetryID              .           
    ## ColorG:SymmetryID              .           
    ## ColorH:SymmetryID              .           
    ## ColorI:SymmetryID              1.889509e-03
    ## ColorE:SymmetryVG             -7.048718e-03
    ## ColorF:SymmetryVG              2.715614e-03
    ## ColorG:SymmetryVG              .           
    ## ColorH:SymmetryVG              6.583228e-03
    ## ColorI:SymmetryVG              .           
    ## ColorE:ReportGIA              -2.750812e-02
    ## ColorF:ReportGIA               .           
    ## ColorG:ReportGIA               .           
    ## ColorH:ReportGIA              -1.777602e-02
    ## ColorI:ReportGIA              -1.080513e-02
    ## PolishG:SymmetryG              .           
    ## PolishID:SymmetryG             .           
    ## PolishVG:SymmetryG             4.820207e-03
    ## PolishG:SymmetryID             .           
    ## PolishID:SymmetryID            2.056885e-04
    ## PolishVG:SymmetryID            .           
    ## PolishG:SymmetryVG             3.248516e-04
    ## PolishID:SymmetryVG            .           
    ## PolishVG:SymmetryVG            1.873860e-02
    ## PolishG:ReportGIA              .           
    ## PolishID:ReportGIA             .           
    ## PolishVG:ReportGIA             .           
    ## SymmetryG:ReportGIA            .           
    ## SymmetryID:ReportGIA           .           
    ## SymmetryVG:ReportGIA           .

``` r
plot(lm.regularized.cv, label = TRUE)
abline(v = minLogLambda)
```

![](diamond-analysis_files/figure-markdown_github/unnamed-chunk-8-1.png)

``` r
lm.regularized <- glmnet(xtrain, ytrain, family = "gaussian", 
                         lambda=lm.regularized.cv$lambda.min)
plot(lm.regularized, xvar = "lambda", label = TRUE)
```

![](diamond-analysis_files/figure-markdown_github/unnamed-chunk-9-1.png)

``` r
lm.regularized.cv.pred <- predict(lm.regularized.cv, newx = xtest, s = "lambda.min") 
lm.regularized.pred <- predict(lm.regularized, newx = xtest, s = "lambda.min") 
head(lm.regularized.cv.pred)
```

    ##              1
    ## 6001  9.834224
    ## 6002 10.705224
    ## 6003  8.193697
    ## 6004 10.101849
    ## 6005  9.631637
    ## 6006  8.983151

``` r
head(lm.regularized.pred)
```

    ##              1
    ## 6001  9.835411
    ## 6002 10.708400
    ## 6003  8.197621
    ## 6004 10.099376
    ## 6005  9.633752
    ## 6006  8.982895

``` r
head(as.numeric(exp(lm.regularized.cv.pred)))
```

    ## [1] 18661.613 44588.191  3618.075 24388.059 15239.364  7967.701

``` r
head(as.numeric(diamond.full.validation$Price))
```

    ## [1]  5169 18609  7666  6224 22241  4238

``` r
accuracy(as.numeric(exp(lm.regularized.cv.pred)), as.numeric(diamond.full.test$Price))
```

    ##                ME     RMSE    MAE        MPE     MAPE
    ## Test set 89.49563 1228.317 666.07 -0.0610813 5.328995

``` r
accuracy(as.numeric(exp(lm.regularized.pred)), as.numeric(diamond.full.test$Price))
```

    ##                ME     RMSE     MAE         MPE     MAPE
    ## Test set 86.61591 1217.089 663.344 -0.06074661 5.328578

Ensemble Forecasts
------------------

Among the methods above, we have identified a few models that yield MAPE less than 11% on the validation set.

``` r
accuracy((exp(bag.tree.pred.valid)), diamond.full.validation$Price)
```

    ##                ME    RMSE      MAE        MPE     MAPE
    ## Test set 54.93479 1145.96 651.0374 -0.3755985 5.467027

``` r
accuracy((exp(random.forest.cv.1.pred.valid)), diamond.full.validation$Price)
```

    ##                ME     RMSE    MAE        MPE     MAPE
    ## Test set 76.09589 1149.842 648.61 -0.3985271 5.407072

``` r
accuracy((exp(boost.cv.pred.valid)), diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE    MAPE
    ## Test set 28.32441 971.3675 539.8008 -0.2647916 4.42206

``` r
accuracy((exp(lm.pred.valid)), diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 40.44794 1148.506 661.4527 -0.5542636 5.512112

``` r
accuracy((exp(lm.step.pred.valid)), diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 40.44794 1148.506 661.4527 -0.5542636 5.512112

``` r
accuracy(as.numeric((exp(lm.regularized.pred.valid))), diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 38.65402 1136.629 657.0175 -0.5803963 5.493791

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(random.forest.cv.1.pred.valid))/2, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 65.51534 1137.684 645.4508 -0.3870628 5.403685

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(boost.cv.pred.valid))/2, diamond.full.validation$Price)
```

    ##               ME     RMSE     MAE       MPE     MAPE
    ## Test set 41.6296 994.7589 565.792 -0.320195 4.719538

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(lm.pred.valid))/2, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 47.69137 1018.192 592.2473 -0.4649311 4.993638

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(lm.step.pred.valid))/2, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 47.69137 1018.192 592.2473 -0.4649311 4.993638

``` r
accuracy(((exp(bag.tree.pred.valid))+as.numeric(exp(lm.regularized.pred.valid)))/2, diamond.full.validation$Price)
```

    ##               ME     RMSE      MAE        MPE     MAPE
    ## Test set 46.7944 1015.909 588.7834 -0.4779974 4.983346

``` r
accuracy(((exp(random.forest.cv.1.pred.valid))+exp(boost.cv.pred.valid))/2, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 52.21015 995.4822 567.3312 -0.3316593 4.701738

``` r
accuracy(((exp(random.forest.cv.1.pred.valid))+exp(lm.pred.valid))/2, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE    MAPE
    ## Test set 58.27192 1028.711 595.6248 -0.4763953 4.98239

``` r
accuracy(((exp(random.forest.cv.1.pred.valid))+exp(lm.step.pred.valid))/2, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE    MAPE
    ## Test set 58.27192 1028.711 595.6248 -0.4763953 4.98239

``` r
accuracy(((exp(random.forest.cv.1.pred.valid))+as.numeric(exp(lm.regularized.pred.valid)))/2, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 57.37495 1029.443 593.2592 -0.4894617 4.977059

``` r
accuracy(((exp(boost.cv.pred.valid))+exp(lm.pred.valid))/2, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE    MAPE
    ## Test set 34.38618 1000.268 572.0866 -0.4095276 4.71909

``` r
accuracy(((exp(boost.cv.pred.valid))+exp(lm.step.pred.valid))/2, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE    MAPE
    ## Test set 34.38618 1000.268 572.0866 -0.4095276 4.71909

``` r
accuracy(((exp(boost.cv.pred.valid))+as.numeric(exp(lm.regularized.pred.valid)))/2, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 33.48921 992.0552 569.3943 -0.4225939 4.712036

``` r
accuracy(((exp(lm.pred.valid))+exp(lm.step.pred.valid))/2, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 40.44794 1148.506 661.4527 -0.5542636 5.512112

``` r
accuracy(((exp(lm.pred.valid))+as.numeric(exp(lm.regularized.pred.valid)))/2, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 39.55098 1140.458 658.2261 -0.5673299 5.493493

``` r
accuracy(((exp(lm.step.pred.valid))+as.numeric(exp(lm.regularized.pred.valid)))/2, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 39.55098 1140.458 658.2261 -0.5673299 5.493493

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(random.forest.cv.1.pred.valid)+exp(boost.cv.pred.valid))/3, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 53.11836 1028.511 587.0046 -0.3463057 4.895199

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(random.forest.cv.1.pred.valid)+exp(lm.pred.valid))/3, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 57.15954 1032.942 597.7987 -0.4427964 5.026136

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(random.forest.cv.1.pred.valid)+exp(lm.step.pred))/3, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE       MPE     MAPE
    ## Test set -29.6834 4724.968 3240.548 -22.44548 38.30009

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(random.forest.cv.1.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/3, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 56.56157 1033.727 596.8723 -0.4515073 5.025347

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(boost.cv.pred.valid)+exp(lm.pred.valid))/3, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 41.23571 973.6155 561.3881 -0.3982179 4.692153

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(boost.cv.pred.valid)+exp(lm.step.pred.valid))/3, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 41.23571 973.6155 561.3881 -0.3982179 4.692153

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(boost.cv.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/3, diamond.full.validation$Price)
```

    ##                ME    RMSE      MAE        MPE     MAPE
    ## Test set 40.63774 970.364 559.7708 -0.4069288 4.689482

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(lm.pred.valid)+exp(lm.step.pred.valid))/3, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 45.27689 1033.797 603.7961 -0.4947086 5.059648

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(lm.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/3, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 44.67892 1030.292 600.5931 -0.5034195 5.049771

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(lm.step.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/3, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 44.67892 1030.292 600.5931 -0.5034195 5.049771

``` r
accuracy(((exp(random.forest.cv.1.pred.valid))+exp(boost.cv.pred.valid)+exp(lm.pred.valid))/3, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 48.28941 978.3384 563.1687 -0.4058608 4.684413

``` r
accuracy(((exp(random.forest.cv.1.pred.valid))+exp(boost.cv.pred.valid)+exp(lm.step.pred.valid))/3, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 48.28941 978.3384 563.1687 -0.4058608 4.684413

``` r
accuracy(((exp(random.forest.cv.1.pred.valid))+exp(boost.cv.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/3, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 47.69144 976.5035 561.3967 -0.4145717 4.683054

``` r
accuracy(((exp(random.forest.cv.1.pred.valid))+exp(lm.pred.valid)+exp(lm.step.pred.valid))/3, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 52.33059 1042.538 604.9404 -0.5023514 5.048649

``` r
accuracy(((exp(random.forest.cv.1.pred.valid))+exp(lm.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/3, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE    MAPE
    ## Test set 51.73262 1040.377 601.7679 -0.5110623 5.03768

``` r
accuracy(((exp(random.forest.cv.1.pred.valid))+exp(lm.step.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/3, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE    MAPE
    ## Test set 51.73262 1040.377 601.7679 -0.5110623 5.03768

``` r
accuracy(((exp(boost.cv.pred.valid))+exp(lm.pred.valid)+exp(lm.step.pred.valid))/3, diamond.full.validation$Price)
```

    ##                ME   RMSE      MAE        MPE     MAPE
    ## Test set 36.40676 1038.1 597.5316 -0.4577729 4.944798

``` r
accuracy(((exp(boost.cv.pred.valid))+exp(lm.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/3, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 35.80879 1032.089 594.5059 -0.4664838 4.931228

``` r
accuracy(((exp(boost.cv.pred.valid))+exp(lm.step.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/3, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 35.80879 1032.089 594.5059 -0.4664838 4.931228

``` r
accuracy(((exp(lm.pred.valid))+exp(lm.step.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/3, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 39.84997 1142.676 659.0316 -0.5629745 5.498411

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(random.forest.cv.1.pred.valid)+exp(boost.cv.pred.valid)+exp(lm.pred.valid))/4, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE    MAPE
    ## Test set 49.95076 990.1215 570.6605 -0.3982952 4.77358

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(random.forest.cv.1.pred.valid)+exp(boost.cv.pred.valid)+exp(lm.step.pred.valid))/4, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE    MAPE
    ## Test set 49.95076 990.1215 570.6605 -0.3982952 4.77358

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(random.forest.cv.1.pred.valid)+exp(boost.cv.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/4, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 49.50228 989.3728 569.8293 -0.4048284 4.771507

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(random.forest.cv.1.pred.valid)+exp(lm.pred.valid)+exp(lm.step.pred.valid))/4, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 52.98164 1020.609 592.0092 -0.4706632 4.974614

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(random.forest.cv.1.pred.valid)+exp(lm.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/4, diamond.full.validation$Price)
```

    ##                ME    RMSE      MAE        MPE     MAPE
    ## Test set 52.53316 1019.63 590.5167 -0.4771964 4.968302

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(random.forest.cv.1.pred.valid)+exp(lm.step.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/4, diamond.full.validation$Price)
```

    ##                ME    RMSE      MAE        MPE     MAPE
    ## Test set 52.53316 1019.63 590.5167 -0.4771964 4.968302

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(boost.cv.pred.valid)+exp(lm.pred.valid)+exp(lm.step.pred.valid))/4, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 41.03877 991.9249 574.9708 -0.4372293 4.798085

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(boost.cv.pred.valid)+exp(lm.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/4, diamond.full.validation$Price)
```

    ##                ME    RMSE      MAE        MPE     MAPE
    ## Test set 40.59029 988.661 572.6283 -0.4437625 4.790984

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(boost.cv.pred.valid)+exp(lm.step.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/4, diamond.full.validation$Price)
```

    ##                ME    RMSE      MAE        MPE     MAPE
    ## Test set 40.59029 988.661 572.6283 -0.4437625 4.790984

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(lm.pred.valid)+exp(lm.step.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/4, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 43.62117 1049.312 611.7683 -0.5161305 5.123025

``` r
accuracy(((exp(random.forest.cv.1.pred.valid))+exp(boost.cv.pred.valid)+exp(lm.pred.valid)+exp(lm.step.pred.valid))/4, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 46.32905 996.9586 575.6492 -0.4429615 4.786447

``` r
accuracy(((exp(random.forest.cv.1.pred.valid))+exp(boost.cv.pred.valid)+exp(lm.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/4, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 45.88056 994.4847 573.9327 -0.4494946 4.781373

``` r
accuracy(((exp(random.forest.cv.1.pred.valid))+exp(boost.cv.pred.valid)+exp(lm.step.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/4, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 45.88056 994.4847 573.9327 -0.4494946 4.781373

``` r
accuracy(((exp(random.forest.cv.1.pred.valid))+exp(lm.pred.valid)+exp(lm.step.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/4, diamond.full.validation$Price)
```

    ##                ME     RMSE    MAE        MPE     MAPE
    ## Test set 48.91145 1057.179 612.74 -0.5218626 5.114418

``` r
accuracy(((exp(boost.cv.pred.valid))+exp(lm.pred.valid)+exp(lm.step.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/4, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 36.96858 1056.977 609.5035 -0.4884288 5.060979

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(random.forest.cv.1.pred.valid)+exp(boost.cv.pred.valid)+exp(lm.pred.valid)+as.numeric(exp(lm.step.pred.valid)))/5, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 48.05019 989.7177 571.7821 -0.4294889 4.785624

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(random.forest.cv.1.pred.valid)+exp(boost.cv.pred.valid)+exp(lm.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/5, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 47.69141 988.2969 570.6465 -0.4347154 4.782432

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(random.forest.cv.1.pred.valid)+exp(boost.cv.pred.valid)+exp(lm.step.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/5, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 47.69141 988.2969 570.6465 -0.4347154 4.782432

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(random.forest.cv.1.pred.valid)+exp(lm.pred.valid)+exp(lm.step.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/5, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 50.11612 1025.465 595.4167 -0.4926098 4.999117

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(boost.cv.pred.valid)+exp(lm.pred.valid)+exp(lm.step.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/5, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE    MAPE
    ## Test set 40.56182 1008.803 585.9314 -0.4658627 4.89129

``` r
accuracy(((exp(random.forest.cv.1.pred.valid))+exp(boost.cv.pred.valid)+exp(lm.pred.valid)+exp(lm.step.pred.valid)+as.numeric(exp(lm.regularized.pred.valid)))/5, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 44.79404 1013.984 586.4854 -0.4704484 4.882159

``` r
accuracy(((exp(bag.tree.pred.valid))+exp(random.forest.cv.1.pred.valid)+exp(boost.cv.pred.valid)+exp(lm.pred.valid)+as.numeric(exp(lm.step.pred.valid))+as.numeric(exp(lm.regularized.pred.valid)))/6, diamond.full.validation$Price)
```

    ##                ME     RMSE      MAE        MPE     MAPE
    ## Test set 46.48416 997.3985 577.3475 -0.4546401 4.832255

Summary of Analysis & Areas for Further Research
================================================

A few key conclusions are worth noting after our analysis of the data:

1.  Best model for predicting diamond prices is a boosted tree model, which gives MAPE of 4.23%

2.  Given the scatter plot between price and carat weight, the log-log relationship makes the most sense

3.  Although we include log(carat weight) and 1/carat weight as explanatory variables, there seem to be distinct clusters of prices based on ranges of carat weights, as the bin dummies based on carat weight we created were m

4.  Though tree-based models tend to outperform linear regression models, they lose out on model intepretability. In particular, it is a lot easier to get an estimate of the marginal effect of a specific attribute on diamond price with a linear regression model vs. a tree-based model.

5.  That said, using a tree-based model to help determine which variables and interaction terms to include as a start in a linear regression appears to be fruitful.
