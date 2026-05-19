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
#> (Intercept)               1.048   0.133   0.792   1.302
#> gender                   -0.512   0.058  -0.633  -0.406
#> as.factor(education)2     0.138   0.129  -0.114   0.379
#> as.factor(education)3    -0.119   0.102  -0.298   0.069
#> as.factor(education)4    -0.411   0.115  -0.645  -0.179
#> as.factor(education)5    -0.537   0.119  -0.766  -0.308
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.856   0.108  -1.057  -0.643
#> gender                    0.477   0.052   0.374   0.579
#> as.factor(education)2    -0.037   0.104  -0.234   0.170
#> as.factor(education)3     0.111   0.083  -0.042   0.282
#> as.factor(education)4    -0.037   0.092  -0.215   0.144
#> as.factor(education)5     0.085   0.096  -0.084   0.280
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.311
#> A2 -0.311  1.000
#> --- 
# }
```
