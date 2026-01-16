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
#>    B1--B2     0.228   0.061   0.100   0.345
#>    B1--B3     0.051   0.069  -0.075   0.194
#>    B2--B3     0.497   0.049   0.389   0.584
#>    B1--B4     0.325   0.056   0.218   0.437
#>    B2--B4    -0.031   0.068  -0.162   0.104
#>    B3--B4     0.222   0.064   0.099   0.342
#>    B1--B5     0.152   0.062   0.037   0.270
#>    B2--B5     0.103   0.073  -0.039   0.233
#>    B3--B5     0.185   0.069   0.048   0.304
#>    B4--B5     0.350   0.060   0.241   0.457
#> --- 

# }
```
