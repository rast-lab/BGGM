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
#> (Intercept)               1.048   0.136   0.828   1.308
#> gender                   -0.513   0.061  -0.639  -0.397
#> as.factor(education)2     0.148   0.111  -0.083   0.361
#> as.factor(education)3    -0.115   0.097  -0.309   0.071
#> as.factor(education)4    -0.414   0.110  -0.615  -0.210
#> as.factor(education)5    -0.532   0.114  -0.733  -0.310
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.858   0.108  -1.054  -0.653
#> gender                    0.483   0.049   0.390   0.573
#> as.factor(education)2    -0.038   0.107  -0.232   0.199
#> as.factor(education)3     0.107   0.083  -0.049   0.272
#> as.factor(education)4    -0.050   0.094  -0.219   0.133
#> as.factor(education)5     0.072   0.099  -0.125   0.263
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.313
#> A2 -0.313  1.000
#> --- 
# }
```
