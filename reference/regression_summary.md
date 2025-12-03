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
#> (Intercept)               1.038   0.126   0.799   1.275
#> gender                   -0.505   0.062  -0.628  -0.396
#> as.factor(education)2     0.146   0.116  -0.075   0.368
#> as.factor(education)3    -0.117   0.100  -0.306   0.069
#> as.factor(education)4    -0.411   0.115  -0.628  -0.186
#> as.factor(education)5    -0.538   0.115  -0.777  -0.332
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.868   0.095  -1.076  -0.676
#> gender                    0.480   0.045   0.368   0.566
#> as.factor(education)2    -0.033   0.103  -0.217   0.167
#> as.factor(education)3     0.118   0.082  -0.026   0.282
#> as.factor(education)4    -0.034   0.098  -0.207   0.157
#> as.factor(education)5     0.081   0.094  -0.097   0.277
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.312
#> A2 -0.312  1.000
#> --- 
# }
```
