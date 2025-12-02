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
#> (Intercept)               1.042   0.132   0.768   1.290
#> gender                   -0.504   0.055  -0.611  -0.389
#> as.factor(education)2     0.133   0.118  -0.082   0.350
#> as.factor(education)3    -0.128   0.097  -0.290   0.072
#> as.factor(education)4    -0.418   0.116  -0.608  -0.181
#> as.factor(education)5    -0.536   0.105  -0.763  -0.332
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.850   0.110  -1.058  -0.625
#> gender                    0.475   0.049   0.376   0.566
#> as.factor(education)2    -0.041   0.107  -0.260   0.159
#> as.factor(education)3     0.111   0.084  -0.050   0.270
#> as.factor(education)4    -0.050   0.097  -0.232   0.151
#> as.factor(education)5     0.072   0.094  -0.109   0.254
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.315
#> A2 -0.315  1.000
#> --- 
# }
```
