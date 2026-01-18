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
#>    B1--B2     0.227   0.066   0.093   0.349
#>    B1--B3     0.049   0.068  -0.067   0.170
#>    B2--B3     0.500   0.053   0.387   0.600
#>    B1--B4     0.334   0.056   0.234   0.443
#>    B2--B4    -0.034   0.070  -0.184   0.099
#>    B3--B4     0.218   0.067   0.085   0.340
#>    B1--B5     0.157   0.065   0.052   0.292
#>    B2--B5     0.106   0.069  -0.031   0.228
#>    B3--B5     0.185   0.066   0.054   0.310
#>    B4--B5     0.338   0.058   0.233   0.450
#> --- 

# }
```
