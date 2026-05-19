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
#>  A1--A3   -0.105    0.017          0.000 0.000 1.000
#>  A2--A3    0.289    0.021          0.000 1.000 0.000
#>  A1--A4   -0.017    0.018          0.996 0.001 0.004
#>  A2--A4    0.159    0.021          0.000 1.000 0.000
#>  A3--A4    0.160    0.019          0.000 1.000 0.000
#>  A1--A5   -0.013    0.019          0.997 0.001 0.002
#>  A2--A5    0.144    0.020          0.000 1.000 0.000
#>  A3--A5    0.355    0.018          0.000 1.000 0.000
#>  A4--A5    0.116    0.019          0.000 1.000 0.000
#>  A1--C1    0.053    0.019          0.182 0.816 0.002
#>  A2--C1    0.006    0.020          0.998 0.001 0.001
#>  A3--C1    0.007    0.021          0.997 0.002 0.001
#>  A4--C1   -0.044    0.018          0.509 0.003 0.488
#>  A5--C1    0.062    0.018          0.004 0.996 0.000
#>  A1--C2    0.071    0.019          0.000 0.999 0.000
#>  A2--C2    0.006    0.019          0.998 0.001 0.001
#>  A3--C2    0.032    0.020          0.971 0.027 0.001
#>  A4--C2    0.151    0.020          0.000 1.000 0.000
#>  A5--C2   -0.028    0.019          0.982 0.001 0.016
#>  C1--C2    0.302    0.021          0.000 1.000 0.000
#>  A1--C3    0.042    0.021          0.867 0.131 0.003
#>  A2--C3    0.125    0.020          0.000 1.000 0.000
#>  A3--C3   -0.014    0.018          0.997 0.001 0.002
#>  A4--C3   -0.031    0.018          0.965 0.001 0.034
#>  A5--C3    0.021    0.021          0.994 0.005 0.001
#>  C1--C3    0.125    0.019          0.000 1.000 0.000
#>  C2--C3    0.181    0.020          0.000 1.000 0.000
#>  A1--C4    0.124    0.018          0.000 1.000 0.000
#>  A2--C4   -0.016    0.020          0.996 0.001 0.003
#>  A3--C4    0.010    0.019          0.997 0.002 0.001
#>  A4--C4    0.017    0.020          0.996 0.003 0.001
#>  A5--C4    0.005    0.021          0.998 0.001 0.001
#>  C1--C4   -0.159    0.019          0.000 0.000 1.000
#>  C2--C4   -0.185    0.020          0.000 0.000 1.000
#>  C3--C4   -0.124    0.020          0.000 0.000 1.000
#>  A1--C5   -0.028    0.019          0.982 0.001 0.017
#>  A2--C5    0.043    0.019          0.753 0.244 0.003
#>  A3--C5   -0.023    0.018          0.991 0.001 0.008
#>  A4--C5   -0.149    0.017          0.000 0.000 1.000
#>  A5--C5   -0.053    0.021          0.351 0.003 0.646
#>  C1--C5   -0.036    0.019          0.940 0.002 0.058
#>  C2--C5   -0.045    0.019          0.650 0.003 0.347
#>  C3--C5   -0.177    0.018          0.000 0.000 1.000
#>  C4--C5    0.360    0.020          0.000 1.000 0.000

# }
```
