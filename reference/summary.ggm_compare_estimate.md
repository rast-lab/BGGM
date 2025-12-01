# Summary method for `ggm_compare_estimate` objects

Summarize the posterior distribution of each partial correlation
difference with the posterior mean and standard deviation.

## Usage

``` r
# S3 method for class 'ggm_compare_estimate'
summary(object, col_names = TRUE, cred = 0.95, ...)
```

## Arguments

- object:

  An object of class `ggm_compare_estimate`.

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

A list containing the summarized posterior distributions.

## See also

[`ggm_compare_estimate`](https://rast-lab.github.io/BGGM/reference/ggm_compare_estimate.md)

## Examples

``` r
# \donttest{
# note: iter = 250 for demonstrative purposes
# data
Y <- bfi

# males and females
Ymale <- subset(Y, gender == 1,
                select = -c(gender,
                            education))[,1:5]

Yfemale <- subset(Y, gender == 2,
                  select = -c(gender,
                              education))[,1:5]

# fit model
fit <- ggm_compare_estimate(Ymale,  Yfemale,
                            type = "continuous",
                            iter = 250,
                            progress = FALSE)

summary(fit)
#> BGGM: Bayesian Gaussian Graphical Models 
#> --- 
#> Type: continuous 
#> Analytic: FALSE 
#> Formula:  
#> Posterior Samples: 250 
#> Observations (n):
#>   Group 1: 896 
#>   Group 2: 1813 
#> Nodes (p): 5 
#> Relations: 10 
#> --- 
#> Call: 
#> ggm_compare_estimate(Ymale, Yfemale, type = "continuous", iter = 250, 
#>     progress = FALSE)
#> --- 
#> Estimates:
#> 
#>  
#>  Relation Post.mean Post.sd Cred.lb Cred.ub
#>  A1--A2    0.047    0.037   -0.026  0.112  
#>  A1--A3   -0.014    0.043   -0.095  0.069  
#>  A2--A3   -0.039    0.039   -0.106  0.041  
#>  A1--A4    0.006    0.038   -0.070  0.079  
#>  A2--A4   -0.001    0.036   -0.075  0.065  
#>  A3--A4    0.019    0.037   -0.058  0.090  
#>  A1--A5    0.006    0.040   -0.069  0.091  
#>  A2--A5    0.076    0.040    0.001  0.142  
#>  A3--A5    0.051    0.036   -0.019  0.121  
#>  A4--A5   -0.009    0.040   -0.078  0.074  
#> --- 
# }
```
