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
#>  A1--A3   -0.107    0.020          0.000 0.000 1.000
#>  A2--A3    0.285    0.020          0.000 1.000 0.000
#>  A1--A4   -0.013    0.020          0.997 0.001 0.003
#>  A2--A4    0.161    0.017          0.000 1.000 0.000
#>  A3--A4    0.161    0.018          0.000 1.000 0.000
#>  A1--A5   -0.015    0.020          0.996 0.001 0.003
#>  A2--A5    0.145    0.021          0.000 1.000 0.000
#>  A3--A5    0.354    0.018          0.000 1.000 0.000
#>  A4--A5    0.114    0.020          0.000 1.000 0.000
#>  A1--C1    0.052    0.019          0.157 0.841 0.002
#>  A2--C1    0.002    0.019          0.998 0.001 0.001
#>  A3--C1    0.008    0.019          0.997 0.002 0.001
#>  A4--C1   -0.046    0.021          0.725 0.003 0.272
#>  A5--C1    0.062    0.020          0.022 0.977 0.001
#>  A1--C2    0.074    0.019          0.000 1.000 0.000
#>  A2--C2    0.010    0.020          0.997 0.002 0.001
#>  A3--C2    0.030    0.019          0.978 0.021 0.001
#>  A4--C2    0.152    0.019          0.000 1.000 0.000
#>  A5--C2   -0.030    0.017          0.964 0.001 0.034
#>  C1--C2    0.302    0.017          0.000 1.000 0.000
#>  A1--C3    0.041    0.018          0.745 0.252 0.003
#>  A2--C3    0.124    0.020          0.000 1.000 0.000
#>  A3--C3   -0.011    0.019          0.997 0.001 0.002
#>  A4--C3   -0.033    0.021          0.970 0.002 0.028
#>  A5--C3    0.023    0.018          0.991 0.008 0.001
#>  C1--C3    0.123    0.019          0.000 1.000 0.000
#>  C2--C3    0.183    0.020          0.000 1.000 0.000
#>  A1--C4    0.123    0.020          0.000 1.000 0.000
#>  A2--C4   -0.016    0.019          0.996 0.001 0.003
#>  A3--C4    0.013    0.019          0.997 0.002 0.001
#>  A4--C4    0.016    0.021          0.995 0.004 0.001
#>  A5--C4    0.003    0.018          0.998 0.001 0.001
#>  C1--C4   -0.157    0.019          0.000 0.000 1.000
#>  C2--C4   -0.189    0.019          0.000 0.000 1.000
#>  C3--C4   -0.126    0.019          0.000 0.000 1.000
#>  A1--C5   -0.025    0.020          0.989 0.001 0.010
#>  A2--C5    0.042    0.020          0.842 0.155 0.003
#>  A3--C5   -0.023    0.019          0.991 0.001 0.008
#>  A4--C5   -0.149    0.019          0.000 0.000 1.000
#>  A5--C5   -0.053    0.019          0.163 0.002 0.835
#>  C1--C5   -0.040    0.018          0.824 0.003 0.173
#>  C2--C5   -0.041    0.018          0.768 0.003 0.229
#>  C3--C5   -0.177    0.019          0.000 0.000 1.000
#>  C4--C5    0.357    0.020          0.000 1.000 0.000

# }
```
