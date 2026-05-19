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
#> (Intercept)               1.049   0.141   0.766   1.319
#> gender                   -0.507   0.058  -0.614  -0.389
#> as.factor(education)2     0.127   0.125  -0.128   0.358
#> as.factor(education)3    -0.129   0.101  -0.314   0.073
#> as.factor(education)4    -0.415   0.110  -0.647  -0.209
#> as.factor(education)5    -0.544   0.119  -0.775  -0.305
#> --- 
#> A2 
#>                       Post.mean Post.sd Cred.lb Cred.ub
#> (Intercept)              -0.864   0.108  -1.055  -0.649
#> gender                    0.480   0.049   0.386   0.573
#> as.factor(education)2    -0.030   0.097  -0.206   0.174
#> as.factor(education)3     0.118   0.079  -0.039   0.280
#> as.factor(education)4    -0.043   0.090  -0.228   0.147
#> as.factor(education)5     0.082   0.090  -0.079   0.262
#> --- 
#> Residual Correlation Matrix: 
#>        A1     A2
#> A1  1.000 -0.313
#> A2 -0.313  1.000
#> --- 
# }
```
