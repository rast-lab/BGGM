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
#>    B1--B2     0.222   0.069   0.087   0.352
#>    B1--B3     0.064   0.067  -0.068   0.188
#>    B2--B3     0.500   0.052   0.391   0.600
#>    B1--B4     0.323   0.057   0.209   0.424
#>    B2--B4    -0.024   0.069  -0.172   0.107
#>    B3--B4     0.213   0.067   0.091   0.332
#>    B1--B5     0.156   0.066   0.022   0.286
#>    B2--B5     0.096   0.068  -0.038   0.221
#>    B3--B5     0.191   0.062   0.068   0.303
#>    B4--B5     0.341   0.061   0.232   0.463
#> --- 

# }
```
