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
#>    B1--B2     0.226   0.063   0.104   0.352
#>    B1--B3     0.061   0.064  -0.065   0.177
#>    B2--B3     0.499   0.051   0.389   0.593
#>    B1--B4     0.335   0.063   0.216   0.458
#>    B2--B4    -0.029   0.066  -0.160   0.104
#>    B3--B4     0.211   0.066   0.076   0.336
#>    B1--B5     0.149   0.062   0.026   0.265
#>    B2--B5     0.106   0.069  -0.032   0.237
#>    B3--B5     0.185   0.065   0.054   0.311
#>    B4--B5     0.342   0.060   0.213   0.443
#> --- 

# }
```
