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
#> (Intercept)               1.044   0.124   0.812   1.292
#> gender                   -0.501   0.058  -0.605  -0.389
#> as.factor(education)2     0.130   0.121  -0.112   0.349
#> as.factor(education)3    -0.136   0.098  -0.328   0.060
#> as.factor(education)4    -0.421   0.110  -0.623  -0.221
#> as.factor(education)5    -0.553   0.108  -0.777  -0.336
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.866   0.109  -1.056  -0.667
#> gender                    0.481   0.048   0.387   0.575
#> as.factor(education)2    -0.040   0.100  -0.224   0.140
#> as.factor(education)3     0.117   0.081  -0.044   0.264
#> as.factor(education)4    -0.041   0.091  -0.236   0.136
#> as.factor(education)5     0.076   0.092  -0.102   0.234
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.315
#> A2 -0.315  1.000
#> --- 
# }
```
