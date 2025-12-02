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
#> (Intercept)               1.065   0.130   0.821   1.308
#> gender                   -0.514   0.058  -0.618  -0.405
#> as.factor(education)2     0.129   0.119  -0.105   0.375
#> as.factor(education)3    -0.128   0.102  -0.338   0.069
#> as.factor(education)4    -0.422   0.109  -0.634  -0.226
#> as.factor(education)5    -0.551   0.110  -0.761  -0.320
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.862   0.111  -1.071  -0.647
#> gender                    0.477   0.051   0.389   0.575
#> as.factor(education)2    -0.028   0.106  -0.222   0.185
#> as.factor(education)3     0.120   0.082  -0.049   0.270
#> as.factor(education)4    -0.034   0.100  -0.245   0.181
#> as.factor(education)5     0.083   0.094  -0.108   0.264
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.313
#> A2 -0.313  1.000
#> --- 
# }
```
