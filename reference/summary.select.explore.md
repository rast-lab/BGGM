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
#>  A1--A2   -0.241    0.019          0.000 0.000 1.000
#>  A1--A3   -0.107    0.018          0.000 0.000 1.000
#>  A2--A3    0.285    0.019          0.000 1.000 0.000
#>  A1--A4   -0.015    0.019          0.997 0.001 0.003
#>  A2--A4    0.160    0.021          0.000 1.000 0.000
#>  A3--A4    0.161    0.020          0.000 1.000 0.000
#>  A1--A5   -0.014    0.018          0.997 0.001 0.002
#>  A2--A5    0.147    0.019          0.000 1.000 0.000
#>  A3--A5    0.356    0.020          0.000 1.000 0.000
#>  A4--A5    0.113    0.020          0.000 1.000 0.000
#>  A1--C1    0.051    0.019          0.306 0.691 0.003
#>  A2--C1    0.005    0.020          0.998 0.001 0.001
#>  A3--C1    0.009    0.021          0.997 0.002 0.001
#>  A4--C1   -0.048    0.020          0.553 0.003 0.444
#>  A5--C1    0.061    0.020          0.064 0.935 0.001
#>  A1--C2    0.074    0.021          0.002 0.998 0.000
#>  A2--C2    0.011    0.019          0.997 0.002 0.001
#>  A3--C2    0.029    0.018          0.977 0.022 0.001
#>  A4--C2    0.153    0.019          0.000 1.000 0.000
#>  A5--C2   -0.027    0.019          0.987 0.001 0.012
#>  C1--C2    0.303    0.019          0.000 1.000 0.000
#>  A1--C3    0.041    0.019          0.845 0.153 0.002
#>  A2--C3    0.125    0.020          0.000 1.000 0.000
#>  A3--C3   -0.012    0.022          0.997 0.001 0.002
#>  A4--C3   -0.033    0.020          0.972 0.001 0.027
#>  A5--C3    0.020    0.020          0.995 0.004 0.001
#>  C1--C3    0.123    0.019          0.000 1.000 0.000
#>  C2--C3    0.184    0.021          0.000 1.000 0.000
#>  A1--C4    0.125    0.020          0.000 1.000 0.000
#>  A2--C4   -0.017    0.020          0.996 0.001 0.003
#>  A3--C4    0.015    0.019          0.997 0.003 0.001
#>  A4--C4    0.014    0.022          0.997 0.003 0.001
#>  A5--C4    0.005    0.019          0.998 0.001 0.001
#>  C1--C4   -0.156    0.022          0.000 0.000 1.000
#>  C2--C4   -0.189    0.020          0.000 0.000 1.000
#>  C3--C4   -0.125    0.020          0.000 0.000 1.000
#>  A1--C5   -0.024    0.021          0.992 0.001 0.007
#>  A2--C5    0.045    0.019          0.710 0.287 0.003
#>  A3--C5   -0.027    0.020          0.988 0.001 0.011
#>  A4--C5   -0.148    0.021          0.000 0.000 1.000
#>  A5--C5   -0.051    0.021          0.511 0.003 0.486
#>  C1--C5   -0.039    0.019          0.901 0.002 0.097
#>  C2--C5   -0.045    0.021          0.816 0.003 0.181
#>  C3--C5   -0.175    0.019          0.000 0.000 1.000
#>  C4--C5    0.356    0.021          0.000 1.000 0.000

# }
```
