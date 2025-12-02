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
#> (Intercept)               1.051   0.129   0.812   1.301
#> gender                   -0.510   0.055  -0.623  -0.416
#> as.factor(education)2     0.144   0.125  -0.108   0.365
#> as.factor(education)3    -0.123   0.096  -0.324   0.057
#> as.factor(education)4    -0.417   0.111  -0.638  -0.199
#> as.factor(education)5    -0.543   0.116  -0.787  -0.343
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.866   0.107  -1.097  -0.673
#> gender                    0.484   0.047   0.395   0.573
#> as.factor(education)2    -0.046   0.102  -0.257   0.140
#> as.factor(education)3     0.112   0.082  -0.044   0.271
#> as.factor(education)4    -0.044   0.097  -0.243   0.144
#> as.factor(education)5     0.078   0.098  -0.108   0.265
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.314
#> A2 -0.314  1.000
#> --- 
# }
```
