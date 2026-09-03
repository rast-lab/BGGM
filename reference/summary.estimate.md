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
#>    B1--B2     0.227   0.063   0.111   0.353
#>    B1--B3     0.065   0.069  -0.054   0.208
#>    B2--B3     0.494   0.049   0.395   0.589
#>    B1--B4     0.330   0.059   0.219   0.434
#>    B2--B4    -0.033   0.065  -0.158   0.092
#>    B3--B4     0.221   0.065   0.100   0.329
#>    B1--B5     0.152   0.069   0.025   0.283
#>    B2--B5     0.111   0.072  -0.034   0.253
#>    B3--B5     0.181   0.068   0.057   0.319
#>    B4--B5     0.338   0.060   0.228   0.445
#> --- 

# }
```
