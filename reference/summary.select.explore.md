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
#>  A1--A3   -0.109    0.020          0.000 0.000 1.000
#>  A2--A3    0.286    0.020          0.000 1.000 0.000
#>  A1--A4   -0.013    0.020          0.997 0.001 0.002
#>  A2--A4    0.162    0.019          0.000 1.000 0.000
#>  A3--A4    0.160    0.020          0.000 1.000 0.000
#>  A1--A5   -0.014    0.022          0.996 0.001 0.003
#>  A2--A5    0.145    0.021          0.000 1.000 0.000
#>  A3--A5    0.354    0.020          0.000 1.000 0.000
#>  A4--A5    0.113    0.020          0.000 1.000 0.000
#>  A1--C1    0.051    0.021          0.563 0.434 0.003
#>  A2--C1    0.004    0.020          0.998 0.001 0.001
#>  A3--C1    0.007    0.019          0.998 0.001 0.001
#>  A4--C1   -0.049    0.020          0.535 0.003 0.462
#>  A5--C1    0.064    0.019          0.011 0.988 0.001
#>  A1--C2    0.074    0.019          0.000 1.000 0.000
#>  A2--C2    0.009    0.020          0.998 0.002 0.001
#>  A3--C2    0.032    0.019          0.970 0.029 0.001
#>  A4--C2    0.154    0.019          0.000 1.000 0.000
#>  A5--C2   -0.031    0.021          0.979 0.001 0.020
#>  C1--C2    0.302    0.021          0.000 1.000 0.000
#>  A1--C3    0.042    0.021          0.883 0.115 0.002
#>  A2--C3    0.125    0.018          0.000 1.000 0.000
#>  A3--C3   -0.016    0.020          0.996 0.001 0.003
#>  A4--C3   -0.027    0.019          0.987 0.001 0.012
#>  A5--C3    0.023    0.018          0.992 0.007 0.001
#>  C1--C3    0.123    0.021          0.000 1.000 0.000
#>  C2--C3    0.183    0.021          0.000 1.000 0.000
#>  A1--C4    0.124    0.021          0.000 1.000 0.000
#>  A2--C4   -0.015    0.019          0.996 0.001 0.003
#>  A3--C4    0.012    0.020          0.997 0.002 0.001
#>  A4--C4    0.016    0.019          0.996 0.003 0.001
#>  A5--C4    0.004    0.019          0.998 0.001 0.001
#>  C1--C4   -0.156    0.021          0.000 0.000 1.000
#>  C2--C4   -0.190    0.020          0.000 0.000 1.000
#>  C3--C4   -0.126    0.022          0.000 0.000 1.000
#>  A1--C5   -0.025    0.019          0.990 0.001 0.009
#>  A2--C5    0.044    0.018          0.606 0.391 0.003
#>  A3--C5   -0.025    0.020          0.991 0.001 0.008
#>  A4--C5   -0.149    0.021          0.000 0.000 1.000
#>  A5--C5   -0.052    0.020          0.403 0.003 0.594
#>  C1--C5   -0.040    0.019          0.852 0.002 0.146
#>  C2--C5   -0.042    0.018          0.705 0.003 0.292
#>  C3--C5   -0.174    0.020          0.000 0.000 1.000
#>  C4--C5    0.358    0.018          0.000 1.000 0.000

# }
```
