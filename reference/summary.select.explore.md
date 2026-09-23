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
#>  A1--A2   -0.243    0.021          0.000 0.000 1.000
#>  A1--A3   -0.106    0.020          0.000 0.000 1.000
#>  A2--A3    0.287    0.021          0.000 1.000 0.000
#>  A1--A4   -0.017    0.019          0.960 0.008 0.033
#>  A2--A4    0.159    0.021          0.000 1.000 0.000
#>  A3--A4    0.159    0.020          0.000 1.000 0.000
#>  A1--A5   -0.015    0.020          0.963 0.008 0.029
#>  A2--A5    0.146    0.019          0.000 1.000 0.000
#>  A3--A5    0.354    0.021          0.000 1.000 0.000
#>  A4--A5    0.114    0.020          0.000 1.000 0.000
#>  A1--C1    0.053    0.020          0.518 0.480 0.002
#>  A2--C1    0.005    0.019          0.972 0.017 0.011
#>  A3--C1    0.008    0.018          0.972 0.019 0.009
#>  A4--C1   -0.046    0.019          0.633 0.002 0.364
#>  A5--C1    0.063    0.019          0.141 0.858 0.000
#>  A1--C2    0.074    0.020          0.031 0.969 0.000
#>  A2--C2    0.010    0.019          0.969 0.021 0.009
#>  A3--C2    0.034    0.020          0.892 0.104 0.005
#>  A4--C2    0.150    0.020          0.000 1.000 0.000
#>  A5--C2   -0.033    0.020          0.897 0.005 0.098
#>  C1--C2    0.299    0.018          0.000 1.000 0.000
#>  A1--C3    0.041    0.018          0.758 0.239 0.003
#>  A2--C3    0.126    0.019          0.000 1.000 0.000
#>  A3--C3   -0.014    0.019          0.965 0.008 0.027
#>  A4--C3   -0.032    0.019          0.894 0.005 0.101
#>  A5--C3    0.020    0.020          0.955 0.038 0.007
#>  C1--C3    0.124    0.020          0.000 1.000 0.000
#>  C2--C3    0.184    0.020          0.000 1.000 0.000
#>  A1--C4    0.124    0.019          0.000 1.000 0.000
#>  A2--C4   -0.015    0.018          0.964 0.007 0.029
#>  A3--C4    0.013    0.021          0.965 0.026 0.010
#>  A4--C4    0.015    0.018          0.964 0.029 0.007
#>  A5--C4    0.004    0.019          0.972 0.016 0.012
#>  C1--C4   -0.157    0.018          0.000 0.000 1.000
#>  C2--C4   -0.189    0.020          0.000 0.000 1.000
#>  C3--C4   -0.121    0.022          0.000 0.000 1.000
#>  A1--C5   -0.027    0.019          0.930 0.006 0.065
#>  A2--C5    0.043    0.018          0.683 0.314 0.003
#>  A3--C5   -0.021    0.020          0.951 0.007 0.042
#>  A4--C5   -0.151    0.019          0.000 0.000 1.000
#>  A5--C5   -0.055    0.019          0.399 0.001 0.600
#>  C1--C5   -0.040    0.019          0.779 0.003 0.218
#>  C2--C5   -0.043    0.019          0.730 0.003 0.267
#>  C3--C5   -0.177    0.019          0.000 0.000 1.000
#>  C4--C5    0.358    0.020          0.000 1.000 0.000

# }
```
