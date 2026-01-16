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
#>  A1--A3   -0.108    0.020          0.000 0.000 1.000
#>  A2--A3    0.286    0.020          0.000 1.000 0.000
#>  A1--A4   -0.014    0.017          0.997 0.001 0.002
#>  A2--A4    0.162    0.019          0.000 1.000 0.000
#>  A3--A4    0.160    0.019          0.000 1.000 0.000
#>  A1--A5   -0.014    0.019          0.997 0.001 0.002
#>  A2--A5    0.144    0.021          0.000 1.000 0.000
#>  A3--A5    0.356    0.021          0.000 1.000 0.000
#>  A4--A5    0.114    0.018          0.000 1.000 0.000
#>  A1--C1    0.053    0.021          0.355 0.642 0.003
#>  A2--C1    0.003    0.020          0.998 0.001 0.001
#>  A3--C1    0.009    0.019          0.998 0.002 0.001
#>  A4--C1   -0.043    0.020          0.820 0.003 0.178
#>  A5--C1    0.060    0.019          0.038 0.961 0.001
#>  A1--C2    0.072    0.020          0.001 0.999 0.000
#>  A2--C2    0.010    0.021          0.997 0.002 0.001
#>  A3--C2    0.031    0.021          0.980 0.019 0.001
#>  A4--C2    0.150    0.019          0.000 1.000 0.000
#>  A5--C2   -0.030    0.018          0.973 0.001 0.025
#>  C1--C2    0.299    0.019          0.000 1.000 0.000
#>  A1--C3    0.039    0.020          0.912 0.086 0.002
#>  A2--C3    0.126    0.020          0.000 1.000 0.000
#>  A3--C3   -0.010    0.018          0.998 0.001 0.002
#>  A4--C3   -0.034    0.019          0.957 0.001 0.041
#>  A5--C3    0.018    0.018          0.996 0.003 0.001
#>  C1--C3    0.124    0.020          0.000 1.000 0.000
#>  C2--C3    0.184    0.019          0.000 1.000 0.000
#>  A1--C4    0.122    0.019          0.000 1.000 0.000
#>  A2--C4   -0.013    0.020          0.997 0.001 0.002
#>  A3--C4    0.011    0.020          0.997 0.002 0.001
#>  A4--C4    0.016    0.020          0.996 0.003 0.001
#>  A5--C4    0.002    0.020          0.998 0.001 0.001
#>  C1--C4   -0.157    0.022          0.000 0.000 1.000
#>  C2--C4   -0.188    0.020          0.000 0.000 1.000
#>  C3--C4   -0.126    0.020          0.000 0.000 1.000
#>  A1--C5   -0.025    0.019          0.990 0.001 0.010
#>  A2--C5    0.045    0.020          0.753 0.244 0.003
#>  A3--C5   -0.024    0.022          0.993 0.001 0.006
#>  A4--C5   -0.148    0.019          0.000 0.000 1.000
#>  A5--C5   -0.053    0.020          0.316 0.003 0.681
#>  C1--C5   -0.040    0.019          0.889 0.002 0.109
#>  C2--C5   -0.043    0.020          0.859 0.003 0.138
#>  C3--C5   -0.175    0.019          0.000 0.000 1.000
#>  C4--C5    0.356    0.019          0.000 1.000 0.000

# }
```
