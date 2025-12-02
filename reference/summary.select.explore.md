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
#>  A1--A2   -0.245    0.020          0.000 0.000 1.000
#>  A1--A3   -0.106    0.018          0.000 0.000 1.000
#>  A2--A3    0.288    0.019          0.000 1.000 0.000
#>  A1--A4   -0.015    0.020          0.996 0.001 0.003
#>  A2--A4    0.161    0.021          0.000 1.000 0.000
#>  A3--A4    0.161    0.021          0.000 1.000 0.000
#>  A1--A5   -0.017    0.019          0.996 0.001 0.003
#>  A2--A5    0.144    0.019          0.000 1.000 0.000
#>  A3--A5    0.353    0.020          0.000 1.000 0.000
#>  A4--A5    0.114    0.021          0.000 1.000 0.000
#>  A1--C1    0.052    0.018          0.134 0.864 0.002
#>  A2--C1    0.003    0.020          0.998 0.001 0.001
#>  A3--C1    0.009    0.019          0.998 0.002 0.001
#>  A4--C1   -0.047    0.020          0.624 0.003 0.373
#>  A5--C1    0.062    0.018          0.006 0.994 0.000
#>  A1--C2    0.072    0.018          0.000 1.000 0.000
#>  A2--C2    0.008    0.019          0.998 0.001 0.001
#>  A3--C2    0.031    0.020          0.976 0.023 0.001
#>  A4--C2    0.151    0.022          0.000 1.000 0.000
#>  A5--C2   -0.030    0.018          0.977 0.001 0.021
#>  C1--C2    0.300    0.020          0.000 1.000 0.000
#>  A1--C3    0.042    0.020          0.847 0.151 0.003
#>  A2--C3    0.126    0.021          0.000 1.000 0.000
#>  A3--C3   -0.013    0.020          0.997 0.001 0.002
#>  A4--C3   -0.034    0.020          0.965 0.002 0.033
#>  A5--C3    0.021    0.020          0.994 0.006 0.001
#>  C1--C3    0.126    0.020          0.000 1.000 0.000
#>  C2--C3    0.185    0.019          0.000 1.000 0.000
#>  A1--C4    0.124    0.018          0.000 1.000 0.000
#>  A2--C4   -0.017    0.019          0.996 0.001 0.003
#>  A3--C4    0.013    0.019          0.997 0.002 0.001
#>  A4--C4    0.016    0.022          0.996 0.003 0.001
#>  A5--C4    0.004    0.020          0.998 0.001 0.001
#>  C1--C4   -0.157    0.021          0.000 0.000 1.000
#>  C2--C4   -0.186    0.021          0.000 0.000 1.000
#>  C3--C4   -0.124    0.019          0.000 0.000 1.000
#>  A1--C5   -0.026    0.019          0.988 0.001 0.011
#>  A2--C5    0.045    0.019          0.667 0.330 0.003
#>  A3--C5   -0.022    0.019          0.993 0.001 0.006
#>  A4--C5   -0.152    0.021          0.000 0.000 1.000
#>  A5--C5   -0.054    0.020          0.270 0.003 0.727
#>  C1--C5   -0.039    0.020          0.922 0.002 0.076
#>  C2--C5   -0.043    0.020          0.848 0.003 0.149
#>  C3--C5   -0.176    0.019          0.000 0.000 1.000
#>  C4--C5    0.357    0.021          0.000 1.000 0.000

# }
```
