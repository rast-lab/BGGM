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
#>  A1--A3   -0.106    0.018          0.000 0.000 1.000
#>  A2--A3    0.285    0.018          0.000 1.000 0.000
#>  A1--A4   -0.016    0.019          0.996 0.001 0.003
#>  A2--A4    0.159    0.020          0.000 1.000 0.000
#>  A3--A4    0.160    0.019          0.000 1.000 0.000
#>  A1--A5   -0.015    0.019          0.996 0.001 0.003
#>  A2--A5    0.145    0.020          0.000 1.000 0.000
#>  A3--A5    0.354    0.021          0.000 1.000 0.000
#>  A4--A5    0.116    0.019          0.000 1.000 0.000
#>  A1--C1    0.055    0.019          0.106 0.892 0.002
#>  A2--C1    0.008    0.020          0.998 0.002 0.001
#>  A3--C1    0.006    0.020          0.998 0.001 0.001
#>  A4--C1   -0.049    0.020          0.518 0.003 0.479
#>  A5--C1    0.063    0.020          0.016 0.983 0.001
#>  A1--C2    0.071    0.020          0.002 0.998 0.000
#>  A2--C2    0.007    0.019          0.998 0.001 0.001
#>  A3--C2    0.031    0.020          0.976 0.023 0.001
#>  A4--C2    0.153    0.019          0.000 1.000 0.000
#>  A5--C2   -0.029    0.020          0.982 0.001 0.017
#>  C1--C2    0.299    0.020          0.000 1.000 0.000
#>  A1--C3    0.041    0.019          0.852 0.145 0.003
#>  A2--C3    0.126    0.020          0.000 1.000 0.000
#>  A3--C3   -0.012    0.020          0.997 0.001 0.002
#>  A4--C3   -0.033    0.020          0.969 0.002 0.030
#>  A5--C3    0.020    0.019          0.995 0.005 0.001
#>  C1--C3    0.127    0.018          0.000 1.000 0.000
#>  C2--C3    0.181    0.019          0.000 1.000 0.000
#>  A1--C4    0.124    0.019          0.000 1.000 0.000
#>  A2--C4   -0.016    0.021          0.996 0.001 0.003
#>  A3--C4    0.011    0.021          0.997 0.002 0.001
#>  A4--C4    0.015    0.018          0.997 0.003 0.001
#>  A5--C4    0.007    0.019          0.998 0.001 0.001
#>  C1--C4   -0.155    0.020          0.000 0.000 1.000
#>  C2--C4   -0.188    0.019          0.000 0.000 1.000
#>  C3--C4   -0.125    0.019          0.000 0.000 1.000
#>  A1--C5   -0.026    0.019          0.988 0.001 0.011
#>  A2--C5    0.044    0.018          0.605 0.392 0.003
#>  A3--C5   -0.024    0.021          0.992 0.001 0.007
#>  A4--C5   -0.153    0.019          0.000 0.000 1.000
#>  A5--C5   -0.052    0.021          0.528 0.003 0.468
#>  C1--C5   -0.042    0.019          0.813 0.003 0.185
#>  C2--C5   -0.044    0.019          0.661 0.003 0.336
#>  C3--C5   -0.176    0.019          0.000 0.000 1.000
#>  C4--C5    0.356    0.019          0.000 1.000 0.000

# }
```
