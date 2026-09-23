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
#>    B1--B2     0.220   0.063   0.097   0.326
#>    B1--B3     0.059   0.064  -0.057   0.184
#>    B2--B3     0.497   0.051   0.396   0.597
#>    B1--B4     0.332   0.059   0.216   0.445
#>    B2--B4    -0.026   0.070  -0.164   0.106
#>    B3--B4     0.217   0.060   0.099   0.325
#>    B1--B5     0.159   0.071   0.016   0.283
#>    B2--B5     0.106   0.067  -0.030   0.223
#>    B3--B5     0.189   0.064   0.057   0.320
#>    B4--B5     0.330   0.062   0.201   0.436
#> --- 

# }
```
