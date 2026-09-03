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
#>  A1--A2   -0.244    0.021          0.000 0.000 1.000
#>  A1--A3   -0.105    0.020          0.000 0.000 1.000
#>  A2--A3    0.287    0.022          0.000 1.000 0.000
#>  A1--A4   -0.014    0.019          0.997 0.001 0.002
#>  A2--A4    0.162    0.019          0.000 1.000 0.000
#>  A3--A4    0.162    0.019          0.000 1.000 0.000
#>  A1--A5   -0.018    0.018          0.996 0.001 0.003
#>  A2--A5    0.146    0.020          0.000 1.000 0.000
#>  A3--A5    0.356    0.019          0.000 1.000 0.000
#>  A4--A5    0.111    0.021          0.000 1.000 0.000
#>  A1--C1    0.052    0.021          0.432 0.564 0.003
#>  A2--C1    0.004    0.019          0.998 0.001 0.001
#>  A3--C1    0.007    0.020          0.998 0.001 0.001
#>  A4--C1   -0.043    0.023          0.923 0.002 0.075
#>  A5--C1    0.062    0.021          0.093 0.906 0.002
#>  A1--C2    0.071    0.019          0.000 0.999 0.000
#>  A2--C2    0.005    0.019          0.998 0.001 0.001
#>  A3--C2    0.033    0.020          0.974 0.024 0.001
#>  A4--C2    0.152    0.020          0.000 1.000 0.000
#>  A5--C2   -0.029    0.019          0.980 0.001 0.019
#>  C1--C2    0.300    0.019          0.000 1.000 0.000
#>  A1--C3    0.043    0.020          0.825 0.172 0.003
#>  A2--C3    0.125    0.018          0.000 1.000 0.000
#>  A3--C3   -0.012    0.019          0.997 0.001 0.002
#>  A4--C3   -0.034    0.020          0.964 0.002 0.034
#>  A5--C3    0.021    0.020          0.994 0.005 0.001
#>  C1--C3    0.123    0.019          0.000 1.000 0.000
#>  C2--C3    0.186    0.020          0.000 1.000 0.000
#>  A1--C4    0.124    0.020          0.000 1.000 0.000
#>  A2--C4   -0.020    0.020          0.995 0.001 0.005
#>  A3--C4    0.012    0.022          0.997 0.002 0.001
#>  A4--C4    0.018    0.019          0.995 0.004 0.001
#>  A5--C4    0.005    0.022          0.997 0.001 0.001
#>  C1--C4   -0.156    0.019          0.000 0.000 1.000
#>  C2--C4   -0.188    0.019          0.000 0.000 1.000
#>  C3--C4   -0.123    0.018          0.000 0.000 1.000
#>  A1--C5   -0.025    0.018          0.989 0.001 0.010
#>  A2--C5    0.042    0.020          0.830 0.167 0.003
#>  A3--C5   -0.022    0.020          0.994 0.001 0.005
#>  A4--C5   -0.151    0.019          0.000 0.000 1.000
#>  A5--C5   -0.053    0.018          0.150 0.002 0.848
#>  C1--C5   -0.038    0.019          0.905 0.002 0.093
#>  C2--C5   -0.045    0.020          0.765 0.003 0.232
#>  C3--C5   -0.174    0.020          0.000 0.000 1.000
#>  C4--C5    0.358    0.018          0.000 1.000 0.000

# }
```
