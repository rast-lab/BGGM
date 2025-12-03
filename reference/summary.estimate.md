# Summary method for `estimate.default` objects

Summarize the posterior distribution of each partial correlation with
the posterior mean and standard deviation.

## Usage

``` r
# S3 method for class 'estimate'
summary(object, col_names = TRUE, cred = 0.95, ...)
```

## Arguments

- object:

  An object of class `estimate`

- col_names:

  Logical. Should the summary include the column names (default is
  `TRUE`)? Setting to `FALSE` includes the column numbers (e.g.,
  `1--2`).

- cred:

  Numeric. The credible interval width for summarizing the posterior
  distributions (defaults to 0.95; must be between 0 and 1).

- ...:

  Currently ignored.

## Value

A dataframe containing the summarized posterior distributions.

## See also

[`estimate`](https://rast-lab.github.io/BGGM/reference/estimate.md)

## Examples

``` r
# \donttest{
# data
Y <- ptsd[,1:5]

fit <- estimate(Y, iter = 250,
                progress = FALSE)

summary(fit)
#> BGGM: Bayesian Gaussian Graphical Models 
#> --- 
#> Type: continuous 
#> Analytic: FALSE 
#> Formula:  
#> Posterior Samples: 250 
#> Observations (n):
#> Nodes (p): 5 
#> Relations: 10 
#> --- 
#> Call: 
#> estimate(Y = Y, iter = 250, progress = FALSE)
#> --- 
#> Estimates:
#>  Relation Post.mean Post.sd Cred.lb Cred.ub
#>    B1--B2     0.230   0.068   0.083   0.350
#>    B1--B3     0.059   0.070  -0.065   0.197
#>    B2--B3     0.496   0.055   0.371   0.594
#>    B1--B4     0.337   0.068   0.190   0.459
#>    B2--B4    -0.042   0.068  -0.171   0.095
#>    B3--B4     0.225   0.062   0.107   0.345
#>    B1--B5     0.144   0.068   0.011   0.282
#>    B2--B5     0.110   0.073  -0.028   0.244
#>    B3--B5     0.183   0.071   0.036   0.305
#>    B4--B5     0.339   0.063   0.205   0.461
#> --- 

# }
```
