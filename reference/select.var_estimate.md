# Graph Selection for `var.estimate` Object

Graph Selection for `var.estimate` Object

## Usage

``` r
# S3 method for class 'var_estimate'
select(object, cred = 0.95, alternative = "two.sided", ...)
```

## Arguments

- object:

  An object of class `VAR.estimate`.

- cred:

  Numeric. The credible interval width for selecting the graph (defaults
  to 0.95; must be between 0 and 1).

- alternative:

  A character string specifying the alternative hypothesis. It must be
  one of "two.sided" (default), "greater" or "less". See note for futher
  details.

- ...:

  Currently ignored.

## Value

An object of class `select.var_estimate`, including

- pcor_adj Adjacency matrix for the partial correlations.

- beta_adj Adjacency matrix for the regression coefficients.

- pcor_weighted_adj Weighted adjacency matrix for the partial
  correlations.

- beta_weighted_adj Weighted adjacency matrix for the regression
  coefficients.

- `pcor_mu` Partial correlation matrix (posterior mean).

- `beta_mu` A matrix including the regression coefficients (posterior
  mean).

## Examples

``` r
# \donttest{
# data
Y <- subset(ifit, id == 1)[,-1]

# fit model with alias (var_estimate also works)
fit <- var_estimate(Y, progress = FALSE)

# select graphs
select(fit, cred = 0.95)
#> BGGM: Bayesian Gaussian Graphical Models 
#> --- 
#> Vector Autoregressive Model (VAR) 
#> --- 
#> Posterior Samples: 5000 
#> Credible Interval: 95 % 
#> --- 
#> Call: 
#> var_estimate(Y = Y, progress = FALSE)
#> --- 
#> Partial Correlations: 
#> 
#>               interested disinterested excited  upset strong stressed steps
#> interested         0.000             0   0.390 -0.202  0.323    0.267     0
#> disinterested      0.000             0   0.000  0.000  0.000    0.000     0
#> excited            0.390             0   0.000  0.000  0.486    0.000     0
#> upset             -0.202             0   0.000  0.000  0.000    0.349     0
#> strong             0.323             0   0.486  0.000  0.000    0.000     0
#> stressed           0.267             0   0.000  0.349  0.000    0.000     0
#> steps              0.000             0   0.000  0.000  0.000    0.000     0
#> --- 
#> Coefficients: 
#> 
#>                  interested disinterested excited upset strong stressed steps
#> interested.l1             0          0.00       0 0.000      0    0.000     0
#> disinterested.l1          0          0.00       0 0.000      0    0.000     0
#> excited.l1                0          0.00       0 0.000      0    0.000     0
#> upset.l1                  0          0.26       0 0.428      0    0.315     0
#> strong.l1                 0          0.00       0 0.000      0    0.000     0
#> stressed.l1               0          0.00       0 0.000      0    0.000     0
#> steps.l1                  0          0.00       0 0.000      0    0.000     0
#> --- 

# }
```
