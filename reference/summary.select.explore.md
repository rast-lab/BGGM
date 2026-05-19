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
#>  A1--A2   -0.246    0.020          0.000 0.000 1.000
#>  A1--A3   -0.104    0.020          0.000 0.000 1.000
#>  A2--A3    0.285    0.019          0.000 1.000 0.000
#>  A1--A4   -0.015    0.020          0.996 0.001 0.003
#>  A2--A4    0.162    0.016          0.000 1.000 0.000
#>  A3--A4    0.159    0.020          0.000 1.000 0.000
#>  A1--A5   -0.015    0.019          0.997 0.001 0.003
#>  A2--A5    0.147    0.020          0.000 1.000 0.000
#>  A3--A5    0.355    0.018          0.000 1.000 0.000
#>  A4--A5    0.113    0.020          0.000 1.000 0.000
#>  A1--C1    0.056    0.019          0.133 0.865 0.002
#>  A2--C1    0.004    0.020          0.998 0.001 0.001
#>  A3--C1    0.009    0.019          0.998 0.002 0.001
#>  A4--C1   -0.047    0.020          0.719 0.003 0.278
#>  A5--C1    0.062    0.021          0.061 0.938 0.001
#>  A1--C2    0.071    0.022          0.008 0.991 0.000
#>  A2--C2    0.007    0.018          0.998 0.001 0.001
#>  A3--C2    0.031    0.018          0.969 0.030 0.001
#>  A4--C2    0.151    0.020          0.000 1.000 0.000
#>  A5--C2   -0.028    0.020          0.985 0.001 0.014
#>  C1--C2    0.302    0.021          0.000 1.000 0.000
#>  A1--C3    0.041    0.020          0.900 0.097 0.002
#>  A2--C3    0.124    0.020          0.000 1.000 0.000
#>  A3--C3   -0.011    0.021          0.997 0.001 0.002
#>  A4--C3   -0.031    0.021          0.982 0.001 0.016
#>  A5--C3    0.018    0.020          0.996 0.004 0.001
#>  C1--C3    0.124    0.021          0.000 1.000 0.000
#>  C2--C3    0.184    0.019          0.000 1.000 0.000
#>  A1--C4    0.124    0.019          0.000 1.000 0.000
#>  A2--C4   -0.017    0.020          0.996 0.001 0.003
#>  A3--C4    0.014    0.021          0.997 0.002 0.001
#>  A4--C4    0.017    0.020          0.996 0.003 0.001
#>  A5--C4    0.006    0.019          0.998 0.001 0.001
#>  C1--C4   -0.159    0.019          0.000 0.000 1.000
#>  C2--C4   -0.187    0.018          0.000 0.000 1.000
#>  C3--C4   -0.125    0.020          0.000 0.000 1.000
#>  A1--C5   -0.026    0.019          0.989 0.001 0.010
#>  A2--C5    0.043    0.020          0.837 0.161 0.003
#>  A3--C5   -0.022    0.019          0.993 0.001 0.006
#>  A4--C5   -0.154    0.019          0.000 0.000 1.000
#>  A5--C5   -0.054    0.020          0.316 0.003 0.681
#>  C1--C5   -0.039    0.021          0.934 0.002 0.064
#>  C2--C5   -0.043    0.017          0.496 0.003 0.502
#>  C3--C5   -0.175    0.019          0.000 0.000 1.000
#>  C4--C5    0.358    0.020          0.000 1.000 0.000

# }
```
