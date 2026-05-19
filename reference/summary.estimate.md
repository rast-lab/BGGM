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
#>    B1--B2     0.229   0.063   0.103   0.341
#>    B1--B3     0.061   0.066  -0.074   0.174
#>    B2--B3     0.488   0.051   0.391   0.577
#>    B1--B4     0.333   0.061   0.217   0.444
#>    B2--B4    -0.032   0.068  -0.170   0.085
#>    B3--B4     0.225   0.067   0.093   0.346
#>    B1--B5     0.147   0.070   0.011   0.281
#>    B2--B5     0.111   0.071  -0.019   0.231
#>    B3--B5     0.182   0.069   0.041   0.307
#>    B4--B5     0.334   0.061   0.203   0.432
#> --- 

# }
```
