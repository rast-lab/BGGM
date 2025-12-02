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
#>    B1--B2     0.221   0.067   0.080   0.343
#>    B1--B3     0.057   0.074  -0.082   0.185
#>    B2--B3     0.499   0.049   0.389   0.585
#>    B1--B4     0.337   0.055   0.232   0.435
#>    B2--B4    -0.028   0.061  -0.154   0.095
#>    B3--B4     0.215   0.067   0.080   0.335
#>    B1--B5     0.154   0.062   0.019   0.276
#>    B2--B5     0.106   0.063  -0.027   0.227
#>    B3--B5     0.188   0.067   0.073   0.319
#>    B4--B5     0.333   0.060   0.226   0.440
#> --- 

# }
```
