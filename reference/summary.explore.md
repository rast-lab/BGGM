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
#>    B1--B2     0.219   0.070
#>    B1--B3     0.061   0.075
#>    B2--B3     0.501   0.053
#>    B1--B4     0.334   0.062
#>    B2--B4    -0.034   0.067
#>    B3--B4     0.216   0.067
#>    B1--B5     0.150   0.066
#>    B2--B5     0.113   0.066
#>    B3--B5     0.175   0.068
#>    B4--B5     0.344   0.055
#> --- 
# }
```
