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
#> (Intercept)               1.066   0.125   0.797   1.301
#> gender                   -0.512   0.055  -0.620  -0.400
#> as.factor(education)2     0.124   0.119  -0.107   0.354
#> as.factor(education)3    -0.136   0.099  -0.337   0.046
#> as.factor(education)4    -0.419   0.119  -0.656  -0.205
#> as.factor(education)5    -0.565   0.110  -0.784  -0.351
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.881   0.100  -1.069  -0.700
#> gender                    0.485   0.044   0.400   0.572
#> as.factor(education)2    -0.026   0.102  -0.225   0.175
#> as.factor(education)3     0.128   0.087  -0.028   0.299
#> as.factor(education)4    -0.032   0.096  -0.209   0.136
#> as.factor(education)5     0.088   0.101  -0.102   0.286
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.312
#> A2 -0.312  1.000
#> --- 
# }
```
