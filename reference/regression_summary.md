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
#> (Intercept)               1.057   0.121   0.824   1.290
#> gender                   -0.512   0.059  -0.614  -0.401
#> as.factor(education)2     0.133   0.123  -0.102   0.374
#> as.factor(education)3    -0.124   0.095  -0.311   0.049
#> as.factor(education)4    -0.415   0.107  -0.627  -0.200
#> as.factor(education)5    -0.543   0.117  -0.758  -0.324
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.866   0.110  -1.094  -0.650
#> gender                    0.480   0.049   0.387   0.576
#> as.factor(education)2    -0.029   0.106  -0.247   0.164
#> as.factor(education)3     0.118   0.085  -0.049   0.281
#> as.factor(education)4    -0.032   0.096  -0.220   0.154
#> as.factor(education)5     0.088   0.098  -0.100   0.266
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.315
#> A2 -0.315  1.000
#> --- 
# }
```
