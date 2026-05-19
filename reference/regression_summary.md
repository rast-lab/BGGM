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
#> (Intercept)               1.053   0.128   0.800   1.291
#> gender                   -0.511   0.060  -0.625  -0.405
#> as.factor(education)2     0.130   0.128  -0.113   0.356
#> as.factor(education)3    -0.128   0.100  -0.300   0.080
#> as.factor(education)4    -0.416   0.114  -0.635  -0.212
#> as.factor(education)5    -0.539   0.116  -0.759  -0.311
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.862   0.110  -1.065  -0.664
#> gender                    0.481   0.047   0.386   0.555
#> as.factor(education)2    -0.037   0.109  -0.252   0.177
#> as.factor(education)3     0.112   0.089  -0.058   0.294
#> as.factor(education)4    -0.043   0.095  -0.209   0.170
#> as.factor(education)5     0.083   0.095  -0.091   0.250
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.312
#> A2 -0.312  1.000
#> --- 
# }
```
