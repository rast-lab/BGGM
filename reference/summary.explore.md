# Summary Method for `explore.default` Objects

Summarize the posterior distribution for each partial correlation with
the posterior mean and standard deviation.

## Usage

``` r
# S3 method for class 'explore'
summary(object, col_names = TRUE, ...)
```

## Arguments

- object:

  An object of class `estimate`

- col_names:

  Logical. Should the summary include the column names (default is
  `TRUE`)? Setting to `FALSE` includes the column numbers (e.g.,
  `1--2`).

- ...:

  Currently ignored

## Value

A dataframe containing the summarized posterior distributions.

## See also

[`select.explore`](https://rast-lab.github.io/BGGM/reference/select.explore.md)

## Examples

``` r
# \donttest{
# note: iter = 250 for demonstrative purposes

Y <- ptsd[,1:5]

fit <- explore(Y, iter = 250,
               progress = FALSE)

summ <- summary(fit)

summ
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
#> explore(Y = Y, iter = 250, progress = FALSE)
#> --- 
#> Estimates:
#>  Relation Post.mean Post.sd
#>    B1--B2     0.228   0.057
#>    B1--B3     0.064   0.071
#>    B2--B3     0.498   0.049
#>    B1--B4     0.329   0.064
#>    B2--B4    -0.033   0.067
#>    B3--B4     0.212   0.066
#>    B1--B5     0.151   0.064
#>    B2--B5     0.109   0.066
#>    B3--B5     0.189   0.068
#>    B4--B5     0.343   0.061
#> --- 
# }
```
