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
#>  A1--A2   -0.243    0.020          0.000 0.000 1.000
#>  A1--A3   -0.106    0.020          0.000 0.000 1.000
#>  A2--A3    0.286    0.020          0.000 1.000 0.000
#>  A1--A4   -0.014    0.020          0.996 0.001 0.003
#>  A2--A4    0.161    0.023          0.000 1.000 0.000
#>  A3--A4    0.162    0.020          0.000 1.000 0.000
#>  A1--A5   -0.015    0.020          0.996 0.001 0.003
#>  A2--A5    0.144    0.018          0.000 1.000 0.000
#>  A3--A5    0.355    0.019          0.000 1.000 0.000
#>  A4--A5    0.112    0.019          0.000 1.000 0.000
#>  A1--C1    0.052    0.020          0.427 0.570 0.003
#>  A2--C1    0.002    0.020          0.998 0.001 0.001
#>  A3--C1    0.008    0.021          0.997 0.002 0.001
#>  A4--C1   -0.045    0.018          0.512 0.003 0.485
#>  A5--C1    0.062    0.020          0.026 0.973 0.001
#>  A1--C2    0.074    0.019          0.000 1.000 0.000
#>  A2--C2    0.010    0.020          0.997 0.002 0.001
#>  A3--C2    0.031    0.021          0.980 0.019 0.001
#>  A4--C2    0.150    0.020          0.000 1.000 0.000
#>  A5--C2   -0.030    0.019          0.976 0.001 0.023
#>  C1--C2    0.300    0.019          0.000 1.000 0.000
#>  A1--C3    0.043    0.019          0.724 0.273 0.003
#>  A2--C3    0.124    0.019          0.000 1.000 0.000
#>  A3--C3   -0.012    0.021          0.997 0.001 0.002
#>  A4--C3   -0.032    0.022          0.979 0.002 0.020
#>  A5--C3    0.020    0.019          0.994 0.005 0.001
#>  C1--C3    0.124    0.020          0.000 1.000 0.000
#>  C2--C3    0.184    0.019          0.000 1.000 0.000
#>  A1--C4    0.125    0.020          0.000 1.000 0.000
#>  A2--C4   -0.016    0.020          0.996 0.001 0.003
#>  A3--C4    0.011    0.019          0.997 0.002 0.001
#>  A4--C4    0.019    0.021          0.995 0.004 0.001
#>  A5--C4    0.004    0.020          0.998 0.001 0.001
#>  C1--C4   -0.156    0.019          0.000 0.000 1.000
#>  C2--C4   -0.189    0.020          0.000 0.000 1.000
#>  C3--C4   -0.124    0.020          0.000 0.000 1.000
#>  A1--C5   -0.025    0.020          0.990 0.001 0.009
#>  A2--C5    0.042    0.020          0.870 0.127 0.003
#>  A3--C5   -0.020    0.019          0.994 0.001 0.005
#>  A4--C5   -0.154    0.019          0.000 0.000 1.000
#>  A5--C5   -0.054    0.017          0.027 0.001 0.973
#>  C1--C5   -0.040    0.018          0.825 0.003 0.172
#>  C2--C5   -0.043    0.018          0.645 0.003 0.352
#>  C3--C5   -0.176    0.019          0.000 0.000 1.000
#>  C4--C5    0.357    0.018          0.000 1.000 0.000

# }
```
