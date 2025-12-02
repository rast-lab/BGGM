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
#> (Intercept)               1.058   0.125   0.819   1.290
#> gender                   -0.513   0.052  -0.602  -0.408
#> as.factor(education)2     0.129   0.123  -0.105   0.358
#> as.factor(education)3    -0.128   0.101  -0.311   0.062
#> as.factor(education)4    -0.409   0.115  -0.660  -0.188
#> as.factor(education)5    -0.542   0.115  -0.771  -0.333
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.862   0.108  -1.064  -0.656
#> gender                    0.483   0.051   0.381   0.583
#> as.factor(education)2    -0.042   0.103  -0.255   0.145
#> as.factor(education)3     0.107   0.083  -0.053   0.270
#> as.factor(education)4    -0.043   0.094  -0.221   0.133
#> as.factor(education)5     0.080   0.091  -0.103   0.245
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.314
#> A2 -0.314  1.000
#> --- 
# }
```
