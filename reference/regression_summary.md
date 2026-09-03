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
#> (Intercept)               1.068   0.119   0.829   1.296
#> gender                   -0.513   0.054  -0.618  -0.412
#> as.factor(education)2     0.130   0.113  -0.101   0.333
#> as.factor(education)3    -0.139   0.092  -0.320   0.020
#> as.factor(education)4    -0.437   0.107  -0.654  -0.247
#> as.factor(education)5    -0.556   0.103  -0.750  -0.363
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.860   0.105  -1.053  -0.656
#> gender                    0.478   0.047   0.393   0.561
#> as.factor(education)2    -0.041   0.097  -0.212   0.127
#> as.factor(education)3     0.120   0.079  -0.025   0.269
#> as.factor(education)4    -0.041   0.091  -0.209   0.134
#> as.factor(education)5     0.079   0.094  -0.095   0.247
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.313
#> A2 -0.313  1.000
#> --- 
# }
```
