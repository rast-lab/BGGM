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
#> (Intercept)               1.039   0.134   0.776   1.285
#> gender                   -0.506   0.061  -0.613  -0.382
#> as.factor(education)2     0.135   0.120  -0.103   0.386
#> as.factor(education)3    -0.119   0.092  -0.293   0.063
#> as.factor(education)4    -0.410   0.112  -0.639  -0.208
#> as.factor(education)5    -0.542   0.107  -0.752  -0.332
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.860   0.103  -1.051  -0.664
#> gender                    0.483   0.049   0.396   0.572
#> as.factor(education)2    -0.042   0.098  -0.254   0.169
#> as.factor(education)3     0.107   0.077  -0.035   0.270
#> as.factor(education)4    -0.047   0.091  -0.226   0.111
#> as.factor(education)5     0.075   0.090  -0.096   0.262
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.314
#> A2 -0.314  1.000
#> --- 
# }
```
