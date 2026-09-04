# Summary Method for `select.explore` Objects

Summary Method for `select.explore` Objects

## Usage

``` r
# S3 method for class 'select.explore'
summary(object, col_names = TRUE, ...)
```

## Arguments

- object:

  object of class `select.explore`.

- col_names:

  Logical.

- ...:

  Currently ignored.

## Value

a data frame including the posterior mean, standard deviation, and
posterior hypothesis probabilities for each relation.

## Examples

``` r
# \donttest{
#  data
Y <- bfi[,1:10]

# fit model
fit <- explore(Y, iter = 250,
               progress = FALSE)

# edge set
E <- select(fit,
            alternative = "exhaustive")

summary(E)
#> BGGM: Bayesian Gaussian Graphical Models 
#> --- 
#> Type: continuous 
#> Alternative: exhaustive 
#> --- 
#> Call:
#> select.explore(object = fit, alternative = "exhaustive")
#> --- 
#> Hypotheses: 
#> H0: rho = 0
#> H1: rho > 0
#> H2: rho < 0 
#> --- 
#> 
#>  Relation Post.mean Post.sd.fisher Pr.H0 Pr.H1 Pr.H2
#>  A1--A2   -0.246    0.019          0.000 0.000 1.000
#>  A1--A3   -0.106    0.019          0.000 0.000 1.000
#>  A2--A3    0.284    0.020          0.000 1.000 0.000
#>  A1--A4   -0.015    0.019          0.996 0.001 0.003
#>  A2--A4    0.163    0.018          0.000 1.000 0.000
#>  A3--A4    0.159    0.019          0.000 1.000 0.000
#>  A1--A5   -0.016    0.021          0.996 0.001 0.003
#>  A2--A5    0.147    0.019          0.000 1.000 0.000
#>  A3--A5    0.354    0.018          0.000 1.000 0.000
#>  A4--A5    0.113    0.020          0.000 1.000 0.000
#>  A1--C1    0.052    0.019          0.261 0.736 0.003
#>  A2--C1    0.003    0.020          0.998 0.001 0.001
#>  A3--C1    0.010    0.019          0.998 0.002 0.001
#>  A4--C1   -0.047    0.022          0.796 0.003 0.200
#>  A5--C1    0.060    0.018          0.006 0.994 0.000
#>  A1--C2    0.076    0.019          0.000 1.000 0.000
#>  A2--C2    0.009    0.018          0.998 0.002 0.001
#>  A3--C2    0.033    0.020          0.966 0.032 0.002
#>  A4--C2    0.151    0.020          0.000 1.000 0.000
#>  A5--C2   -0.031    0.018          0.967 0.001 0.031
#>  C1--C2    0.299    0.017          0.000 1.000 0.000
#>  A1--C3    0.042    0.020          0.857 0.140 0.003
#>  A2--C3    0.123    0.020          0.000 1.000 0.000
#>  A3--C3   -0.013    0.021          0.997 0.001 0.002
#>  A4--C3   -0.033    0.017          0.939 0.002 0.059
#>  A5--C3    0.021    0.020          0.993 0.006 0.001
#>  C1--C3    0.127    0.019          0.000 1.000 0.000
#>  C2--C3    0.182    0.019          0.000 1.000 0.000
#>  A1--C4    0.124    0.019          0.000 1.000 0.000
#>  A2--C4   -0.020    0.019          0.995 0.001 0.005
#>  A3--C4    0.012    0.018          0.997 0.002 0.001
#>  A4--C4    0.018    0.020          0.995 0.004 0.001
#>  A5--C4    0.006    0.018          0.998 0.001 0.001
#>  C1--C4   -0.158    0.020          0.000 0.000 1.000
#>  C2--C4   -0.190    0.021          0.000 0.000 1.000
#>  C3--C4   -0.122    0.021          0.000 0.000 1.000
#>  A1--C5   -0.027    0.020          0.988 0.001 0.011
#>  A2--C5    0.046    0.018          0.531 0.466 0.003
#>  A3--C5   -0.022    0.019          0.993 0.001 0.006
#>  A4--C5   -0.153    0.021          0.000 0.000 1.000
#>  A5--C5   -0.057    0.019          0.048 0.001 0.951
#>  C1--C5   -0.041    0.020          0.884 0.002 0.114
#>  C2--C5   -0.044    0.019          0.731 0.003 0.266
#>  C3--C5   -0.177    0.018          0.000 0.000 1.000
#>  C4--C5    0.357    0.019          0.000 1.000 0.000

# }
```
