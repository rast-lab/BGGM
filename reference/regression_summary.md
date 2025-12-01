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
#> (Intercept)               1.048   0.135   0.815   1.296
#> gender                   -0.511   0.060  -0.637  -0.398
#> as.factor(education)2     0.143   0.123  -0.114   0.367
#> as.factor(education)3    -0.119   0.103  -0.306   0.066
#> as.factor(education)4    -0.409   0.109  -0.615  -0.210
#> as.factor(education)5    -0.544   0.116  -0.794  -0.328
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.860   0.111  -1.080  -0.648
#> gender                    0.481   0.050   0.380   0.583
#> as.factor(education)2    -0.040   0.094  -0.223   0.135
#> as.factor(education)3     0.111   0.081  -0.047   0.257
#> as.factor(education)4    -0.051   0.094  -0.222   0.132
#> as.factor(education)5     0.081   0.092  -0.082   0.263
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.315
#> A2 -0.315  1.000
#> --- 
# }
```
