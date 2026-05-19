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
#>    B1--B2     0.225   0.067   0.099   0.346
#>    B1--B3     0.056   0.064  -0.069   0.172
#>    B2--B3     0.501   0.048   0.401   0.584
#>    B1--B4     0.331   0.061   0.207   0.445
#>    B2--B4    -0.029   0.065  -0.154   0.098
#>    B3--B4     0.218   0.064   0.090   0.330
#>    B1--B5     0.158   0.062   0.029   0.268
#>    B2--B5     0.104   0.070  -0.024   0.227
#>    B3--B5     0.181   0.070   0.046   0.323
#>    B4--B5     0.331   0.062   0.212   0.439
#> --- 

# }
```
