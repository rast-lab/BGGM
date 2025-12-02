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
#>    B1--B2     0.224   0.067   0.104   0.347
#>    B1--B3     0.060   0.066  -0.075   0.181
#>    B2--B3     0.492   0.049   0.389   0.581
#>    B1--B4     0.328   0.063   0.195   0.444
#>    B2--B4    -0.026   0.068  -0.163   0.095
#>    B3--B4     0.214   0.074   0.071   0.368
#>    B1--B5     0.160   0.064   0.047   0.282
#>    B2--B5     0.104   0.059   0.001   0.222
#>    B3--B5     0.188   0.065   0.051   0.298
#>    B4--B5     0.340   0.063   0.222   0.465
#> --- 

# }
```
