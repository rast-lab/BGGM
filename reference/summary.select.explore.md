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
#>  A1--A2   -0.244    0.022          0.000 0.000 1.000
#>  A1--A3   -0.104    0.021          0.000 0.000 1.000
#>  A2--A3    0.288    0.020          0.000 1.000 0.000
#>  A1--A4   -0.014    0.020          0.996 0.001 0.003
#>  A2--A4    0.160    0.021          0.000 1.000 0.000
#>  A3--A4    0.162    0.021          0.000 1.000 0.000
#>  A1--A5   -0.017    0.021          0.996 0.001 0.004
#>  A2--A5    0.145    0.020          0.000 1.000 0.000
#>  A3--A5    0.351    0.020          0.000 1.000 0.000
#>  A4--A5    0.114    0.020          0.000 1.000 0.000
#>  A1--C1    0.053    0.021          0.457 0.539 0.003
#>  A2--C1    0.005    0.020          0.998 0.001 0.001
#>  A3--C1    0.008    0.019          0.998 0.002 0.001
#>  A4--C1   -0.048    0.020          0.570 0.003 0.427
#>  A5--C1    0.063    0.020          0.028 0.971 0.001
#>  A1--C2    0.071    0.019          0.001 0.999 0.000
#>  A2--C2    0.008    0.019          0.998 0.001 0.001
#>  A3--C2    0.031    0.020          0.979 0.020 0.001
#>  A4--C2    0.150    0.020          0.000 1.000 0.000
#>  A5--C2   -0.030    0.018          0.975 0.001 0.024
#>  C1--C2    0.302    0.020          0.000 1.000 0.000
#>  A1--C3    0.041    0.020          0.889 0.109 0.002
#>  A2--C3    0.124    0.020          0.000 1.000 0.000
#>  A3--C3   -0.014    0.020          0.997 0.001 0.003
#>  A4--C3   -0.033    0.017          0.948 0.002 0.050
#>  A5--C3    0.024    0.019          0.991 0.008 0.001
#>  C1--C3    0.126    0.021          0.000 1.000 0.000
#>  C2--C3    0.180    0.018          0.000 1.000 0.000
#>  A1--C4    0.122    0.021          0.000 1.000 0.000
#>  A2--C4   -0.017    0.020          0.996 0.001 0.004
#>  A3--C4    0.010    0.020          0.997 0.002 0.001
#>  A4--C4    0.015    0.019          0.997 0.003 0.001
#>  A5--C4    0.006    0.021          0.997 0.002 0.001
#>  C1--C4   -0.155    0.020          0.000 0.000 1.000
#>  C2--C4   -0.189    0.020          0.000 0.000 1.000
#>  C3--C4   -0.125    0.018          0.000 0.000 1.000
#>  A1--C5   -0.026    0.020          0.989 0.001 0.010
#>  A2--C5    0.044    0.020          0.797 0.200 0.003
#>  A3--C5   -0.022    0.020          0.993 0.001 0.006
#>  A4--C5   -0.153    0.020          0.000 0.000 1.000
#>  A5--C5   -0.053    0.021          0.412 0.003 0.585
#>  C1--C5   -0.038    0.020          0.929 0.002 0.069
#>  C2--C5   -0.045    0.020          0.762 0.003 0.235
#>  C3--C5   -0.176    0.020          0.000 0.000 1.000
#>  C4--C5    0.357    0.021          0.000 1.000 0.000

# }
```
