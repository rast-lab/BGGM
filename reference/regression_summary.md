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
#> (Intercept)               1.050   0.122   0.812   1.286
#> gender                   -0.505   0.055  -0.607  -0.409
#> as.factor(education)2     0.118   0.126  -0.120   0.387
#> as.factor(education)3    -0.136   0.099  -0.330   0.088
#> as.factor(education)4    -0.420   0.115  -0.647  -0.175
#> as.factor(education)5    -0.557   0.117  -0.784  -0.317
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.877   0.102  -1.076  -0.667
#> gender                    0.482   0.049   0.395   0.576
#> as.factor(education)2    -0.022   0.097  -0.207   0.164
#> as.factor(education)3     0.127   0.079  -0.020   0.299
#> as.factor(education)4    -0.028   0.096  -0.227   0.146
#> as.factor(education)5     0.093   0.095  -0.102   0.273
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.314
#> A2 -0.314  1.000
#> --- 
# }
```
