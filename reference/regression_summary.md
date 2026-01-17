# Summarary Method for Multivariate or Univarate Regression

Summarary Method for Multivariate or Univarate Regression

## Usage

``` r
regression_summary(object, cred = 0.95, ...)
```

## Arguments

- object:

  An object of class `estimate`

- cred:

  Numeric. The credible interval width for summarizing the posterior
  distributions (defaults to 0.95; must be between 0 and 1).

- ...:

  Currently ignored

## Value

A list of length *p* including the summaries for each regression.

## Examples

``` r
# \donttest{
# note: iter = 250 for demonstrative purposes

# data
Y <- bfi

Y <- subset(Y, select = c("A1", "A2", 
                          "gender", "education"))

fit_mv_ordinal <- estimate(Y, formula = ~ gender + as.factor(education),
                           type = "continuous",
                           iter = 250,
                           progress = TRUE)
#> BGGM: Posterior Sampling 
#> BGGM: Finished

regression_summary(fit_mv_ordinal)
#> BGGM: Bayesian Gaussian Graphical Models 
#> --- 
#> Type: continuous 
#> Formula: ~ gender + as.factor(education) 
#> --- 
#> Coefficients: 
#>  
#> A1 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)               1.048   0.124   0.818   1.292
#> gender                   -0.510   0.060  -0.646  -0.404
#> as.factor(education)2     0.142   0.122  -0.101   0.353
#> as.factor(education)3    -0.121   0.099  -0.323   0.073
#> as.factor(education)4    -0.419   0.110  -0.625  -0.202
#> as.factor(education)5    -0.541   0.114  -0.767  -0.302
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.871   0.101  -1.087  -0.681
#> gender                    0.485   0.045   0.402   0.574
#> as.factor(education)2    -0.032   0.100  -0.217   0.156
#> as.factor(education)3     0.118   0.081  -0.031   0.261
#> as.factor(education)4    -0.043   0.094  -0.240   0.117
#> as.factor(education)5     0.081   0.094  -0.097   0.264
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.312
#> A2 -0.312  1.000
#> --- 
# }
```
