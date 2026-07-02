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
#>    B1--B2     0.230   0.065
#>    B1--B3     0.056   0.071
#>    B2--B3     0.496   0.052
#>    B1--B4     0.337   0.064
#>    B2--B4    -0.039   0.069
#>    B3--B4     0.221   0.063
#>    B1--B5     0.155   0.060
#>    B2--B5     0.106   0.067
#>    B3--B5     0.181   0.063
#>    B4--B5     0.335   0.058
#> --- 
# }
```
