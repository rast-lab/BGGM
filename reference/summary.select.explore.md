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
#>  A1--A2   -0.245    0.018          0.000 0.000 1.000
#>  A1--A3   -0.108    0.019          0.000 0.000 1.000
#>  A2--A3    0.286    0.019          0.000 1.000 0.000
#>  A1--A4   -0.012    0.018          0.997 0.001 0.002
#>  A2--A4    0.163    0.019          0.000 1.000 0.000
#>  A3--A4    0.159    0.020          0.000 1.000 0.000
#>  A1--A5   -0.013    0.019          0.997 0.001 0.002
#>  A2--A5    0.144    0.019          0.000 1.000 0.000
#>  A3--A5    0.356    0.020          0.000 1.000 0.000
#>  A4--A5    0.115    0.022          0.000 1.000 0.000
#>  A1--C1    0.055    0.019          0.092 0.907 0.002
#>  A2--C1    0.003    0.018          0.998 0.001 0.001
#>  A3--C1    0.010    0.020          0.997 0.002 0.001
#>  A4--C1   -0.050    0.019          0.362 0.003 0.635
#>  A5--C1    0.062    0.018          0.004 0.996 0.000
#>  A1--C2    0.072    0.018          0.000 1.000 0.000
#>  A2--C2    0.009    0.021          0.997 0.002 0.001
#>  A3--C2    0.030    0.020          0.980 0.019 0.001
#>  A4--C2    0.152    0.019          0.000 1.000 0.000
#>  A5--C2   -0.030    0.020          0.980 0.001 0.018
#>  C1--C2    0.304    0.019          0.000 1.000 0.000
#>  A1--C3    0.042    0.019          0.792 0.205 0.003
#>  A2--C3    0.127    0.019          0.000 1.000 0.000
#>  A3--C3   -0.013    0.020          0.997 0.001 0.002
#>  A4--C3   -0.032    0.018          0.963 0.002 0.036
#>  A5--C3    0.021    0.020          0.994 0.005 0.001
#>  C1--C3    0.125    0.019          0.000 1.000 0.000
#>  C2--C3    0.182    0.019          0.000 1.000 0.000
#>  A1--C4    0.123    0.020          0.000 1.000 0.000
#>  A2--C4   -0.018    0.020          0.995 0.001 0.004
#>  A3--C4    0.013    0.019          0.997 0.002 0.001
#>  A4--C4    0.015    0.020          0.996 0.003 0.001
#>  A5--C4    0.006    0.020          0.998 0.001 0.001
#>  C1--C4   -0.156    0.018          0.000 0.000 1.000
#>  C2--C4   -0.188    0.020          0.000 0.000 1.000
#>  C3--C4   -0.125    0.018          0.000 0.000 1.000
#>  A1--C5   -0.024    0.019          0.990 0.001 0.009
#>  A2--C5    0.045    0.020          0.771 0.226 0.003
#>  A3--C5   -0.022    0.020          0.993 0.001 0.006
#>  A4--C5   -0.150    0.021          0.000 0.000 1.000
#>  A5--C5   -0.054    0.020          0.258 0.003 0.739
#>  C1--C5   -0.038    0.019          0.913 0.002 0.084
#>  C2--C5   -0.043    0.018          0.672 0.003 0.325
#>  C3--C5   -0.177    0.021          0.000 0.000 1.000
#>  C4--C5    0.357    0.019          0.000 1.000 0.000

# }
```
