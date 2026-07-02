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
#> (Intercept)               1.050   0.128   0.784   1.327
#> gender                   -0.507   0.060  -0.619  -0.388
#> as.factor(education)2     0.131   0.123  -0.112   0.367
#> as.factor(education)3    -0.127   0.097  -0.332   0.051
#> as.factor(education)4    -0.417   0.115  -0.632  -0.187
#> as.factor(education)5    -0.548   0.112  -0.803  -0.349
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.864   0.110  -1.080  -0.664
#> gender                    0.482   0.046   0.400   0.570
#> as.factor(education)2    -0.035   0.113  -0.262   0.196
#> as.factor(education)3     0.114   0.085  -0.043   0.281
#> as.factor(education)4    -0.041   0.102  -0.247   0.155
#> as.factor(education)5     0.082   0.092  -0.112   0.256
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.313
#> A2 -0.313  1.000
#> --- 
# }
```
