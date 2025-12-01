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
#>    B1--B2     0.224   0.060   0.107   0.335
#>    B1--B3     0.060   0.063  -0.065   0.172
#>    B2--B3     0.496   0.052   0.393   0.595
#>    B1--B4     0.331   0.060   0.216   0.464
#>    B2--B4    -0.031   0.062  -0.143   0.085
#>    B3--B4     0.225   0.060   0.117   0.336
#>    B1--B5     0.153   0.070   0.021   0.279
#>    B2--B5     0.113   0.071  -0.030   0.245
#>    B3--B5     0.176   0.066   0.054   0.306
#>    B4--B5     0.343   0.063   0.212   0.459
#> --- 

# }
```
