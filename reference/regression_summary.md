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
#> (Intercept)               1.050   0.138   0.792   1.318
#> gender                   -0.505   0.059  -0.616  -0.376
#> as.factor(education)2     0.133   0.118  -0.103   0.359
#> as.factor(education)3    -0.125   0.099  -0.324   0.046
#> as.factor(education)4    -0.419   0.117  -0.644  -0.192
#> as.factor(education)5    -0.549   0.120  -0.811  -0.331
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.862   0.103  -1.046  -0.655
#> gender                    0.480   0.050   0.381   0.566
#> as.factor(education)2    -0.039   0.098  -0.205   0.178
#> as.factor(education)3     0.117   0.080  -0.052   0.263
#> as.factor(education)4    -0.040   0.092  -0.216   0.127
#> as.factor(education)5     0.082   0.097  -0.103   0.264
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.313
#> A2 -0.313  1.000
#> --- 
# }
```
