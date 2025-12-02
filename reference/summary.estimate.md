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
#>    B1--B2     0.225   0.063   0.105   0.344
#>    B1--B3     0.060   0.069  -0.069   0.198
#>    B2--B3     0.498   0.052   0.405   0.587
#>    B1--B4     0.335   0.058   0.216   0.445
#>    B2--B4    -0.030   0.072  -0.172   0.102
#>    B3--B4     0.222   0.071   0.080   0.357
#>    B1--B5     0.154   0.069   0.033   0.279
#>    B2--B5     0.105   0.064  -0.021   0.210
#>    B3--B5     0.186   0.069   0.055   0.315
#>    B4--B5     0.333   0.062   0.225   0.443
#> --- 

# }
```
