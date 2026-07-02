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
#>  A1--A3   -0.106    0.020          0.000 0.000 1.000
#>  A2--A3    0.286    0.019          0.000 1.000 0.000
#>  A1--A4   -0.016    0.022          0.996 0.001 0.003
#>  A2--A4    0.162    0.020          0.000 1.000 0.000
#>  A3--A4    0.161    0.021          0.000 1.000 0.000
#>  A1--A5   -0.017    0.020          0.996 0.001 0.003
#>  A2--A5    0.145    0.019          0.000 1.000 0.000
#>  A3--A5    0.353    0.019          0.000 1.000 0.000
#>  A4--A5    0.113    0.018          0.000 1.000 0.000
#>  A1--C1    0.053    0.020          0.339 0.658 0.003
#>  A2--C1    0.005    0.020          0.998 0.001 0.001
#>  A3--C1    0.008    0.019          0.998 0.001 0.001
#>  A4--C1   -0.048    0.020          0.642 0.003 0.355
#>  A5--C1    0.065    0.019          0.007 0.992 0.000
#>  A1--C2    0.070    0.020          0.004 0.996 0.000
#>  A2--C2    0.007    0.021          0.998 0.001 0.001
#>  A3--C2    0.033    0.020          0.973 0.026 0.001
#>  A4--C2    0.149    0.019          0.000 1.000 0.000
#>  A5--C2   -0.029    0.020          0.985 0.001 0.014
#>  C1--C2    0.303    0.020          0.000 1.000 0.000
#>  A1--C3    0.040    0.019          0.886 0.112 0.002
#>  A2--C3    0.126    0.020          0.000 1.000 0.000
#>  A3--C3   -0.014    0.022          0.997 0.001 0.002
#>  A4--C3   -0.031    0.020          0.981 0.001 0.018
#>  A5--C3    0.021    0.020          0.995 0.005 0.001
#>  C1--C3    0.125    0.019          0.000 1.000 0.000
#>  C2--C3    0.183    0.019          0.000 1.000 0.000
#>  A1--C4    0.124    0.020          0.000 1.000 0.000
#>  A2--C4   -0.015    0.019          0.997 0.001 0.002
#>  A3--C4    0.011    0.020          0.998 0.002 0.001
#>  A4--C4    0.017    0.019          0.997 0.003 0.001
#>  A5--C4    0.008    0.018          0.998 0.001 0.001
#>  C1--C4   -0.157    0.019          0.000 0.000 1.000
#>  C2--C4   -0.187    0.020          0.000 0.000 1.000
#>  C3--C4   -0.125    0.021          0.000 0.000 1.000
#>  A1--C5   -0.026    0.017          0.985 0.001 0.014
#>  A2--C5    0.044    0.018          0.627 0.370 0.003
#>  A3--C5   -0.022    0.018          0.994 0.001 0.005
#>  A4--C5   -0.151    0.020          0.000 0.000 1.000
#>  A5--C5   -0.055    0.019          0.106 0.002 0.892
#>  C1--C5   -0.037    0.019          0.938 0.002 0.060
#>  C2--C5   -0.044    0.021          0.834 0.003 0.164
#>  C3--C5   -0.175    0.020          0.000 0.000 1.000
#>  C4--C5    0.359    0.018          0.000 1.000 0.000

# }
```
