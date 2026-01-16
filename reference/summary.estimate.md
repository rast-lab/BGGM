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
#>    B1--B2     0.223   0.067   0.060   0.345
#>    B1--B3     0.060   0.069  -0.078   0.203
#>    B2--B3     0.494   0.055   0.394   0.589
#>    B1--B4     0.330   0.061   0.200   0.460
#>    B2--B4    -0.029   0.066  -0.148   0.105
#>    B3--B4     0.224   0.065   0.117   0.354
#>    B1--B5     0.158   0.066   0.029   0.281
#>    B2--B5     0.105   0.066  -0.018   0.220
#>    B3--B5     0.179   0.065   0.043   0.300
#>    B4--B5     0.334   0.063   0.216   0.475
#> --- 

# }
```
