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
#> (Intercept)               1.061   0.127   0.797   1.300
#> gender                   -0.511   0.059  -0.626  -0.395
#> as.factor(education)2     0.135   0.118  -0.081   0.368
#> as.factor(education)3    -0.130   0.094  -0.288   0.060
#> as.factor(education)4    -0.431   0.108  -0.637  -0.224
#> as.factor(education)5    -0.552   0.116  -0.771  -0.332
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.865   0.108  -1.076  -0.667
#> gender                    0.482   0.047   0.389   0.571
#> as.factor(education)2    -0.043   0.105  -0.249   0.155
#> as.factor(education)3     0.112   0.086  -0.049   0.276
#> as.factor(education)4    -0.037   0.099  -0.220   0.178
#> as.factor(education)5     0.085   0.095  -0.085   0.269
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.314
#> A2 -0.314  1.000
#> --- 
# }
```
