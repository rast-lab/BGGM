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
#> (Intercept)               1.043   0.134   0.791   1.285
#> gender                   -0.502   0.059  -0.627  -0.396
#> as.factor(education)2     0.126   0.126  -0.125   0.365
#> as.factor(education)3    -0.130   0.104  -0.350   0.066
#> as.factor(education)4    -0.421   0.114  -0.632  -0.205
#> as.factor(education)5    -0.553   0.108  -0.769  -0.358
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.871   0.109  -1.081  -0.675
#> gender                    0.478   0.050   0.382   0.568
#> as.factor(education)2    -0.020   0.100  -0.210   0.196
#> as.factor(education)3     0.130   0.082  -0.031   0.314
#> as.factor(education)4    -0.025   0.094  -0.198   0.145
#> as.factor(education)5     0.099   0.094  -0.090   0.300
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.315
#> A2 -0.315  1.000
#> --- 
# }
```
