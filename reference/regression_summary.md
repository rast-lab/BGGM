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
#> (Intercept)               1.066   0.127   0.783   1.303
#> gender                   -0.519   0.061  -0.628  -0.403
#> as.factor(education)2     0.138   0.118  -0.139   0.362
#> as.factor(education)3    -0.124   0.100  -0.316   0.063
#> as.factor(education)4    -0.407   0.113  -0.605  -0.177
#> as.factor(education)5    -0.553   0.110  -0.747  -0.298
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.873   0.107  -1.090  -0.657
#> gender                    0.483   0.047   0.399   0.576
#> as.factor(education)2    -0.030   0.101  -0.277   0.161
#> as.factor(education)3     0.122   0.081  -0.028   0.264
#> as.factor(education)4    -0.039   0.095  -0.215   0.148
#> as.factor(education)5     0.086   0.094  -0.083   0.258
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.312
#> A2 -0.312  1.000
#> --- 
# }
```
