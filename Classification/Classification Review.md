Classification Review
================
Fan Gong
2017/7/5

Classification
==============

Preview
-------

In terms of classification problems, we want solve the quesiton of estimating *P**r*(*Y* = *k*|*X* = *x*<sub>0</sub>).

-   For logistic regression, we assign this probability to logistic function. The *linear* decision boundary is easy to find when *P* = 0.5.

-   For Linear Discriminant Analysis, we use bayes theorem to make $Pr(Y=k|X=x\_0)=\\frac{\\pi\_kf\_k(x\_0)}{\\sum\_{l=1}^{K}\\pi\_lf\_l(x\_0)}$ and assume *f*<sub>*k*</sub>(*x*<sub>0</sub>) belongs to normal distribution. So we approximates the Bayes classifier by plugging estimates for *π*<sub>*k*</sub>, *μ*<sub>*k*</sub> and *σ*<sup>2</sup> into the above equation. The *linear* decision boundary is easy to find when *P**r*(*Y* = 1|*X* = *x*<sub>0</sub>)=*P**r*(*Y* = 0|*X* = *x*<sub>0</sub>).

-   For Quadratic Discriminant Analysis, we also assume it belongs to normal distribution but it allows each class has its own cavariance matrix.

-   For the k-nearest neighbors algorithm, it is a non-parametric method.

Logistic Regression
-------------------

Logistic Regression involves directly modeling *P**r*(*Y* = *k*|*X* = *x*) using the logistic function.

-   The logistic function is:

$p(x) = Pr(Y=1|X)=\\frac{e^{\\beta\_0+\\beta\_1X}}{1+e^{\\beta0+\\beta1X}}$

-   Log-odds or logit:

$log(\\frac{p(x)}{1-p(x)})=\\beta\_0+\\beta\_1X$

If the log-odds larger than zero, we assign it to class one. If the log-odds smaller than zero, we assign it to class two. That is the reason we deem logistic regression as linear classifier.

For *multiple logistic regression*, $log(\\frac{p(x)}{1-p(x)})=\\beta\_0+\\beta\_1X\_1+\\cdots+\\beta\_pX\_p$

The multiple-class logistic regression models are rarely used since discriminant analysis is more popular. Usually people will use two-class logistic regression.

``` r
library(ISLR)
attach(Smarket)
head(Smarket)
```

    ##   Year   Lag1   Lag2   Lag3   Lag4   Lag5 Volume  Today Direction
    ## 1 2001  0.381 -0.192 -2.624 -1.055  5.010 1.1913  0.959        Up
    ## 2 2001  0.959  0.381 -0.192 -2.624 -1.055 1.2965  1.032        Up
    ## 3 2001  1.032  0.959  0.381 -0.192 -2.624 1.4112 -0.623      Down
    ## 4 2001 -0.623  1.032  0.959  0.381 -0.192 1.2760  0.614        Up
    ## 5 2001  0.614 -0.623  1.032  0.959  0.381 1.2057  0.213        Up
    ## 6 2001  0.213  0.614 -0.623  1.032  0.959 1.3491  1.392        Up

``` r
#data pre-processing
train = Smarket[Smarket$Year<2005,]
test = Smarket[Smarket$Year == 2005,]

glm.fit = glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume, family = binomial, data = train)
summary(glm.fit)
```

    ## 
    ## Call:
    ## glm(formula = Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + 
    ##     Volume, family = binomial, data = train)
    ## 
    ## Deviance Residuals: 
    ##    Min      1Q  Median      3Q     Max  
    ## -1.302  -1.190   1.079   1.160   1.350  
    ## 
    ## Coefficients:
    ##              Estimate Std. Error z value Pr(>|z|)
    ## (Intercept)  0.191213   0.333690   0.573    0.567
    ## Lag1        -0.054178   0.051785  -1.046    0.295
    ## Lag2        -0.045805   0.051797  -0.884    0.377
    ## Lag3         0.007200   0.051644   0.139    0.889
    ## Lag4         0.006441   0.051706   0.125    0.901
    ## Lag5        -0.004223   0.051138  -0.083    0.934
    ## Volume      -0.116257   0.239618  -0.485    0.628
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1383.3  on 997  degrees of freedom
    ## Residual deviance: 1381.1  on 991  degrees of freedom
    ## AIC: 1395.1
    ## 
    ## Number of Fisher Scoring iterations: 3

``` r
#Make a prediction by using test data
glm.probs = predict(glm.fit, test, type = "response")
glm.pred = rep("Down", nrow(test))
glm.pred[glm.probs > 0.5] = "Up"

#test error rate
mean(glm.pred != test$Direction)
```

    ## [1] 0.5198413

``` r
#The output is disappointing, we could only use lag1 to try again since it has the lowest p-value.

glm.fit2 = glm(Direction ~ Lag1 + Lag2, family = binomial, data = train)
summary(glm.fit2)
```

    ## 
    ## Call:
    ## glm(formula = Direction ~ Lag1 + Lag2, family = binomial, data = train)
    ## 
    ## Deviance Residuals: 
    ##    Min      1Q  Median      3Q     Max  
    ## -1.345  -1.188   1.074   1.164   1.326  
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)
    ## (Intercept)  0.03222    0.06338   0.508    0.611
    ## Lag1        -0.05562    0.05171  -1.076    0.282
    ## Lag2        -0.04449    0.05166  -0.861    0.389
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1383.3  on 997  degrees of freedom
    ## Residual deviance: 1381.4  on 995  degrees of freedom
    ## AIC: 1387.4
    ## 
    ## Number of Fisher Scoring iterations: 3

``` r
glm.probs2 = predict(glm.fit2, test, type = "response")
glm.pred2 = rep("Down", nrow(test))
glm.pred2[glm.probs2 > 0.5] = "Up"
mean(glm.pred2 != test$Direction)
```

    ## [1] 0.4404762

Linear Discriminant Analysis
----------------------------

### Comparison with logistic regression

-   When the classes are well-separated, the parameter estimates for the logistic regression model are surprisingly unstable.
-   If n is small and the distribution of the predictors X is approximately normal in each of the classes, the LDA model more stable.
-   When we have more than two response classes, LDA is more popular.

### Using Bayes' Theorem

To solve the quesiton of estimating *P**r*(*Y* = *k*|*X* = *x*<sub>0</sub>), the Bayes' theorem states that

$Pr(Y=k|X=x\_0)=\\frac{\\pi\_kf\_k(x\_0)}{\\sum\_{l=1}^{K}\\pi\_lf\_l(x\_0)}$. Here $\\pi\_k = p(Y=k)= \\frac{\\{\\\#i, y\_i=k\\}}{n}$; *f*<sub>*k*</sub>(*x*)=*p*(*X* = *x*<sub>0</sub>|*Y* = *k*) is the thing we want to estimate. Since if we have this estimation ,we could get *P**r*(*Y* = *k*|*X* = *x*<sub>0</sub>). To achieve this goal, we need to assume some simple forms for these densities.

For LDA, we will assume *f*<sub>*k*</sub>(*x*) belongs to multivariate normal distribution.

``` r
#we still use the Smarket data
library(MASS)

lda.fit = lda(Direction ~ Lag1 + Lag2, data = train)

#The Coefficients of linear discriminants provides the linear combination of Lag1 and Lag2. If -0.642*Lag1-0.514*Lag2 is large, then the LDA classifier will predict a incease, and vice versa.
lda.fit
```

    ## Call:
    ## lda(Direction ~ Lag1 + Lag2, data = train)
    ## 
    ## Prior probabilities of groups:
    ##     Down       Up 
    ## 0.491984 0.508016 
    ## 
    ## Group means:
    ##             Lag1        Lag2
    ## Down  0.04279022  0.03389409
    ## Up   -0.03954635 -0.03132544
    ## 
    ## Coefficients of linear discriminants:
    ##             LD1
    ## Lag1 -0.6420190
    ## Lag2 -0.5135293

``` r
#Make a prediction
##The first element is class. Second is a matrix whose kth column contains the posterior probability that the corresponding observation belongs to the kth class. x contains the linear discriminants.
lda.pred = predict(lda.fit, test)
lda.class = lda.pred$class
table(lda.class, test$Direction)
```

    ##          
    ## lda.class Down  Up
    ##      Down   35  35
    ##      Up     76 106

``` r
#Test error
mean(lda.class == test$Direction)
```

    ## [1] 0.5595238

``` r
#If we want to use a posterior probability threshold other than 50% in order to make a predictions, we could do like that:
sum(lda.pred$posterior[,1] > 0.51)
```

    ## [1] 10

Quadratic Discriminant Analysis
-------------------------------

### Comparison with LDA

-   LDA assumes that the observations within each class are drawn from a multivariate Gaussian distribution with a class-specific mean vector and a convariance matrix that is common to all k classes. QDA also assume normal dis but it allows each class has its own cavariance matrix.

-   LDA is a much less flexible classifier than QDA(has fewer parameters), and so has lower variance. So, LDA tends to be a better bet than QDA if there are few observations and reducing variance is crucial. In contrast, QDA is recommended if the training set is very large, so that the variance is not a major concern.

``` r
#syntax is identical to lda
qda.fit = qda(Direction ~ Lag1 + Lag2, data = train)
qda.fit
```

    ## Call:
    ## qda(Direction ~ Lag1 + Lag2, data = train)
    ## 
    ## Prior probabilities of groups:
    ##     Down       Up 
    ## 0.491984 0.508016 
    ## 
    ## Group means:
    ##             Lag1        Lag2
    ## Down  0.04279022  0.03389409
    ## Up   -0.03954635 -0.03132544

``` r
#make a prediction
qda.class = predict(qda.fit, test)$class
mean(qda.class == test$Direction)
```

    ## [1] 0.5992063

K-Nearest Neighbors
-------------------

`knn()` function requires four inputs 1. A matrix containing the predictors associated with the training data. 2. A matrix containing the predictors associated with the data for which we wish to make predictions 3. A vector containing the class labels for the training observations 4. A value for K, the number of nearest neighbors to be used by the classifier.

``` r
library(class)
#data pre-processing
train.X = data.frame(Lag1 = train$Lag1, Lag2 = train$Lag2)
test.X = data.frame(Lag1 = test$Lag1, Lag2 = test$Lag2)
train.Direction = train$Direction

#make a prediction
set.seed(1)
knn.pred = knn(train.X, test.X, train.Direction, k = 1)

mean(knn.pred == test$Direction)
```

    ## [1] 0.5

``` r
#try differenent K
knn.pred2 = knn(train.X, test.X, train.Direction, k = 3)
mean(knn.pred2 == test$Direction)
```

    ## [1] 0.5357143

A Comparison of Classification Methods
--------------------------------------

-   LDA and Logistic regression are quite similar, they differ only in their fitting procedures.

-   KNN is a non-parametric approach, it performs better when the decision boundary is highly non-linear.

-   QDA serves as a compromise between the KNN and LDA/logistic
