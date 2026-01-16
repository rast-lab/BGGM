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
#>    B1--B2     0.229   0.067   0.093   0.367
#>    B1--B3     0.059   0.061  -0.065   0.172
#>    B2--B3     0.497   0.051   0.401   0.583
#>    B1--B4     0.328   0.064   0.215   0.446
#>    B2--B4    -0.036   0.068  -0.154   0.103
#>    B3--B4     0.226   0.061   0.114   0.343
#>    B1--B5     0.154   0.071   0.021   0.293
#>    B2--B5     0.112   0.069  -0.021   0.230
#>    B3--B5     0.173   0.072   0.033   0.328
#>    B4--B5     0.332   0.064   0.207   0.452
#> --- 

# }
```
