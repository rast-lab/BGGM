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
#> (Intercept)               1.062   0.130   0.820   1.305
#> gender                   -0.511   0.059  -0.617  -0.396
#> as.factor(education)2     0.114   0.128  -0.122   0.351
#> as.factor(education)3    -0.133   0.100  -0.305   0.058
#> as.factor(education)4    -0.423   0.117  -0.659  -0.213
#> as.factor(education)5    -0.557   0.121  -0.806  -0.330
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.865   0.104  -1.051  -0.669
#> gender                    0.480   0.043   0.397   0.560
#> as.factor(education)2    -0.028   0.101  -0.239   0.152
#> as.factor(education)3     0.118   0.088  -0.040   0.293
#> as.factor(education)4    -0.042   0.099  -0.243   0.144
#> as.factor(education)5     0.086   0.097  -0.107   0.278
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.314
#> A2 -0.314  1.000
#> --- 
# }
```
