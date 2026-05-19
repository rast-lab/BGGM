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
#>  A1--A2   -0.243    0.019          0.000 0.000 1.000
#>  A1--A3   -0.105    0.020          0.000 0.000 1.000
#>  A2--A3    0.286    0.021          0.000 1.000 0.000
#>  A1--A4   -0.016    0.021          0.996 0.001 0.003
#>  A2--A4    0.159    0.019          0.000 1.000 0.000
#>  A3--A4    0.159    0.019          0.000 1.000 0.000
#>  A1--A5   -0.016    0.020          0.996 0.001 0.003
#>  A2--A5    0.145    0.019          0.000 1.000 0.000
#>  A3--A5    0.355    0.019          0.000 1.000 0.000
#>  A4--A5    0.115    0.019          0.000 1.000 0.000
#>  A1--C1    0.054    0.020          0.206 0.792 0.002
#>  A2--C1    0.004    0.018          0.998 0.001 0.001
#>  A3--C1    0.010    0.021          0.997 0.002 0.001
#>  A4--C1   -0.047    0.020          0.665 0.003 0.332
#>  A5--C1    0.060    0.020          0.063 0.936 0.001
#>  A1--C2    0.070    0.019          0.001 0.999 0.000
#>  A2--C2    0.008    0.019          0.998 0.001 0.001
#>  A3--C2    0.032    0.019          0.973 0.026 0.001
#>  A4--C2    0.150    0.021          0.000 1.000 0.000
#>  A5--C2   -0.030    0.019          0.979 0.001 0.020
#>  C1--C2    0.300    0.019          0.000 1.000 0.000
#>  A1--C3    0.040    0.020          0.897 0.101 0.002
#>  A2--C3    0.124    0.020          0.000 1.000 0.000
#>  A3--C3   -0.012    0.021          0.997 0.001 0.002
#>  A4--C3   -0.032    0.020          0.977 0.001 0.022
#>  A5--C3    0.020    0.020          0.995 0.005 0.001
#>  C1--C3    0.126    0.018          0.000 1.000 0.000
#>  C2--C3    0.182    0.019          0.000 1.000 0.000
#>  A1--C4    0.125    0.019          0.000 1.000 0.000
#>  A2--C4   -0.017    0.021          0.996 0.001 0.003
#>  A3--C4    0.012    0.021          0.997 0.002 0.001
#>  A4--C4    0.015    0.020          0.997 0.003 0.001
#>  A5--C4    0.006    0.018          0.998 0.001 0.001
#>  C1--C4   -0.158    0.020          0.000 0.000 1.000
#>  C2--C4   -0.190    0.021          0.000 0.000 1.000
#>  C3--C4   -0.125    0.019          0.000 0.000 1.000
#>  A1--C5   -0.027    0.021          0.988 0.001 0.011
#>  A2--C5    0.041    0.018          0.807 0.191 0.002
#>  A3--C5   -0.021    0.019          0.994 0.001 0.005
#>  A4--C5   -0.149    0.021          0.000 0.000 1.000
#>  A5--C5   -0.056    0.018          0.039 0.001 0.961
#>  C1--C5   -0.039    0.020          0.929 0.002 0.069
#>  C2--C5   -0.043    0.019          0.780 0.003 0.217
#>  C3--C5   -0.176    0.018          0.000 0.000 1.000
#>  C4--C5    0.357    0.021          0.000 1.000 0.000

# }
```
