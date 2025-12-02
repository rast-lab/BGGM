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
#>    B1--B2     0.217   0.062   0.103   0.336
#>    B1--B3     0.066   0.067  -0.057   0.209
#>    B2--B3     0.497   0.052   0.391   0.585
#>    B1--B4     0.338   0.062   0.200   0.453
#>    B2--B4    -0.028   0.069  -0.162   0.117
#>    B3--B4     0.211   0.067   0.079   0.334
#>    B1--B5     0.155   0.064   0.032   0.274
#>    B2--B5     0.108   0.066  -0.015   0.232
#>    B3--B5     0.186   0.064   0.046   0.308
#>    B4--B5     0.331   0.062   0.211   0.455
#> --- 

# }
```
