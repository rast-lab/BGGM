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
#>  A1--A2   -0.245    0.019          0.000 0.000 1.000
#>  A1--A3   -0.106    0.022          0.000 0.000 1.000
#>  A2--A3    0.286    0.021          0.000 1.000 0.000
#>  A1--A4   -0.016    0.021          0.996 0.001 0.003
#>  A2--A4    0.158    0.019          0.000 1.000 0.000
#>  A3--A4    0.162    0.020          0.000 1.000 0.000
#>  A1--A5   -0.017    0.020          0.996 0.001 0.003
#>  A2--A5    0.143    0.021          0.000 1.000 0.000
#>  A3--A5    0.354    0.021          0.000 1.000 0.000
#>  A4--A5    0.116    0.020          0.000 1.000 0.000
#>  A1--C1    0.051    0.020          0.364 0.633 0.003
#>  A2--C1    0.004    0.018          0.998 0.001 0.001
#>  A3--C1    0.008    0.019          0.998 0.001 0.001
#>  A4--C1   -0.047    0.021          0.742 0.003 0.255
#>  A5--C1    0.063    0.019          0.009 0.991 0.000
#>  A1--C2    0.073    0.021          0.003 0.997 0.000
#>  A2--C2    0.009    0.018          0.998 0.001 0.001
#>  A3--C2    0.032    0.021          0.980 0.019 0.001
#>  A4--C2    0.151    0.020          0.000 1.000 0.000
#>  A5--C2   -0.033    0.020          0.971 0.001 0.027
#>  C1--C2    0.303    0.019          0.000 1.000 0.000
#>  A1--C3    0.043    0.021          0.888 0.109 0.002
#>  A2--C3    0.123    0.019          0.000 1.000 0.000
#>  A3--C3   -0.010    0.021          0.997 0.001 0.002
#>  A4--C3   -0.030    0.018          0.976 0.001 0.023
#>  A5--C3    0.020    0.020          0.995 0.005 0.001
#>  C1--C3    0.125    0.021          0.000 1.000 0.000
#>  C2--C3    0.183    0.020          0.000 1.000 0.000
#>  A1--C4    0.124    0.020          0.000 1.000 0.000
#>  A2--C4   -0.017    0.020          0.996 0.001 0.003
#>  A3--C4    0.012    0.020          0.997 0.002 0.001
#>  A4--C4    0.016    0.020          0.996 0.003 0.001
#>  A5--C4    0.003    0.020          0.998 0.001 0.001
#>  C1--C4   -0.157    0.019          0.000 0.000 1.000
#>  C2--C4   -0.188    0.019          0.000 0.000 1.000
#>  C3--C4   -0.125    0.019          0.000 0.000 1.000
#>  A1--C5   -0.029    0.018          0.979 0.001 0.020
#>  A2--C5    0.040    0.019          0.881 0.116 0.002
#>  A3--C5   -0.022    0.020          0.993 0.001 0.006
#>  A4--C5   -0.148    0.018          0.000 0.000 1.000
#>  A5--C5   -0.055    0.019          0.109 0.002 0.889
#>  C1--C5   -0.038    0.020          0.940 0.002 0.058
#>  C2--C5   -0.044    0.018          0.661 0.003 0.336
#>  C3--C5   -0.175    0.017          0.000 0.000 1.000
#>  C4--C5    0.357    0.017          0.000 1.000 0.000

# }
```
