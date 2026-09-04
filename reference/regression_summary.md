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
#> (Intercept)               1.059   0.128   0.803   1.294
#> gender                   -0.512   0.055  -0.615  -0.406
#> as.factor(education)2     0.142   0.128  -0.131   0.392
#> as.factor(education)3    -0.127   0.097  -0.295   0.062
#> as.factor(education)4    -0.424   0.108  -0.622  -0.223
#> as.factor(education)5    -0.542   0.105  -0.734  -0.335
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.855   0.112  -1.088  -0.644
#> gender                    0.475   0.046   0.387   0.570
#> as.factor(education)2    -0.036   0.103  -0.233   0.155
#> as.factor(education)3     0.115   0.079  -0.033   0.267
#> as.factor(education)4    -0.040   0.101  -0.238   0.157
#> as.factor(education)5     0.080   0.091  -0.090   0.250
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.313
#> A2 -0.313  1.000
#> --- 
# }
```
