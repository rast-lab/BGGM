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
#>  A1--A2   -0.246    0.019          0.000 0.000 1.000
#>  A1--A3   -0.106    0.018          0.000 0.000 1.000
#>  A2--A3    0.284    0.019          0.000 1.000 0.000
#>  A1--A4   -0.013    0.020          0.997 0.001 0.002
#>  A2--A4    0.160    0.018          0.000 1.000 0.000
#>  A3--A4    0.162    0.017          0.000 1.000 0.000
#>  A1--A5   -0.015    0.020          0.997 0.001 0.003
#>  A2--A5    0.146    0.021          0.000 1.000 0.000
#>  A3--A5    0.353    0.020          0.000 1.000 0.000
#>  A4--A5    0.116    0.020          0.000 1.000 0.000
#>  A1--C1    0.053    0.019          0.170 0.828 0.002
#>  A2--C1    0.003    0.019          0.998 0.001 0.001
#>  A3--C1    0.010    0.019          0.998 0.002 0.001
#>  A4--C1   -0.047    0.018          0.437 0.003 0.561
#>  A5--C1    0.064    0.019          0.007 0.993 0.000
#>  A1--C2    0.072    0.019          0.000 1.000 0.000
#>  A2--C2    0.007    0.020          0.998 0.001 0.001
#>  A3--C2    0.032    0.019          0.967 0.032 0.001
#>  A4--C2    0.151    0.019          0.000 1.000 0.000
#>  A5--C2   -0.029    0.020          0.983 0.001 0.016
#>  C1--C2    0.301    0.019          0.000 1.000 0.000
#>  A1--C3    0.043    0.019          0.808 0.189 0.003
#>  A2--C3    0.126    0.019          0.000 1.000 0.000
#>  A3--C3   -0.013    0.020          0.997 0.001 0.002
#>  A4--C3   -0.031    0.020          0.977 0.001 0.022
#>  A5--C3    0.017    0.020          0.996 0.003 0.001
#>  C1--C3    0.125    0.019          0.000 1.000 0.000
#>  C2--C3    0.185    0.020          0.000 1.000 0.000
#>  A1--C4    0.124    0.019          0.000 1.000 0.000
#>  A2--C4   -0.016    0.018          0.997 0.001 0.003
#>  A3--C4    0.014    0.018          0.997 0.002 0.001
#>  A4--C4    0.018    0.020          0.996 0.004 0.001
#>  A5--C4    0.004    0.019          0.998 0.001 0.001
#>  C1--C4   -0.157    0.021          0.000 0.000 1.000
#>  C2--C4   -0.188    0.021          0.000 0.000 1.000
#>  C3--C4   -0.125    0.019          0.000 0.000 1.000
#>  A1--C5   -0.024    0.019          0.991 0.001 0.008
#>  A2--C5    0.042    0.019          0.837 0.160 0.003
#>  A3--C5   -0.022    0.020          0.993 0.001 0.006
#>  A4--C5   -0.153    0.022          0.000 0.000 1.000
#>  A5--C5   -0.053    0.020          0.340 0.003 0.657
#>  C1--C5   -0.039    0.019          0.898 0.002 0.100
#>  C2--C5   -0.042    0.019          0.818 0.003 0.179
#>  C3--C5   -0.174    0.020          0.000 0.000 1.000
#>  C4--C5    0.357    0.019          0.000 1.000 0.000

# }
```
