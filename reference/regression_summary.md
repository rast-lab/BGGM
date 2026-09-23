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
#> (Intercept)               1.041   0.134   0.796   1.296
#> gender                   -0.510   0.060  -0.622  -0.412
#> as.factor(education)2     0.145   0.123  -0.093   0.390
#> as.factor(education)3    -0.112   0.103  -0.294   0.098
#> as.factor(education)4    -0.402   0.119  -0.622  -0.164
#> as.factor(education)5    -0.537   0.109  -0.755  -0.325
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.851   0.110  -1.063  -0.620
#> gender                    0.477   0.051   0.383   0.571
#> as.factor(education)2    -0.036   0.102  -0.226   0.160
#> as.factor(education)3     0.110   0.085  -0.047   0.281
#> as.factor(education)4    -0.053   0.095  -0.231   0.153
#> as.factor(education)5     0.079   0.090  -0.090   0.276
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.314
#> A2 -0.314  1.000
#> --- 
# }
```
