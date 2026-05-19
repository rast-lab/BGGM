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
#>    B1--B2     0.230   0.067   0.102   0.361
#>    B1--B3     0.055   0.077  -0.095   0.197
#>    B2--B3     0.499   0.051   0.394   0.589
#>    B1--B4     0.330   0.060   0.210   0.444
#>    B2--B4    -0.032   0.065  -0.161   0.093
#>    B3--B4     0.219   0.061   0.100   0.339
#>    B1--B5     0.150   0.068   0.029   0.283
#>    B2--B5     0.099   0.071  -0.032   0.231
#>    B3--B5     0.193   0.064   0.070   0.316
#>    B4--B5     0.339   0.058   0.226   0.449
#> --- 

# }
```
