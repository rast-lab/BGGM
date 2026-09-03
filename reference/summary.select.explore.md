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
#>  A1--A2   -0.242    0.021          0.000 0.000 1.000
#>  A1--A3   -0.106    0.018          0.000 0.000 1.000
#>  A2--A3    0.287    0.020          0.000 1.000 0.000
#>  A1--A4   -0.016    0.020          0.924 0.016 0.060
#>  A2--A4    0.161    0.018          0.000 1.000 0.000
#>  A3--A4    0.159    0.019          0.000 1.000 0.000
#>  A1--A5   -0.016    0.020          0.923 0.015 0.061
#>  A2--A5    0.145    0.019          0.000 1.000 0.000
#>  A3--A5    0.353    0.021          0.000 1.000 0.000
#>  A4--A5    0.115    0.020          0.000 1.000 0.000
#>  A1--C1    0.049    0.018          0.339 0.658 0.003
#>  A2--C1    0.007    0.018          0.945 0.036 0.019
#>  A3--C1    0.006    0.020          0.942 0.036 0.023
#>  A4--C1   -0.047    0.018          0.411 0.003 0.585
#>  A5--C1    0.062    0.019          0.094 0.905 0.001
#>  A1--C2    0.072    0.021          0.036 0.964 0.000
#>  A2--C2    0.006    0.021          0.939 0.038 0.023
#>  A3--C2    0.034    0.018          0.748 0.245 0.007
#>  A4--C2    0.149    0.019          0.000 1.000 0.000
#>  A5--C2   -0.031    0.019          0.812 0.009 0.179
#>  C1--C2    0.301    0.019          0.000 1.000 0.000
#>  A1--C3    0.044    0.021          0.639 0.354 0.006
#>  A2--C3    0.126    0.021          0.000 1.000 0.000
#>  A3--C3   -0.012    0.020          0.932 0.019 0.049
#>  A4--C3   -0.031    0.020          0.841 0.010 0.149
#>  A5--C3    0.019    0.019          0.914 0.073 0.014
#>  C1--C3    0.124    0.020          0.000 1.000 0.000
#>  C2--C3    0.183    0.018          0.000 1.000 0.000
#>  A1--C4    0.125    0.020          0.000 1.000 0.000
#>  A2--C4   -0.019    0.020          0.915 0.015 0.070
#>  A3--C4    0.014    0.021          0.928 0.054 0.018
#>  A4--C4    0.018    0.020          0.919 0.066 0.015
#>  A5--C4    0.005    0.019          0.944 0.035 0.022
#>  C1--C4   -0.158    0.021          0.000 0.000 1.000
#>  C2--C4   -0.187    0.018          0.000 0.000 1.000
#>  C3--C4   -0.125    0.019          0.000 0.000 1.000
#>  A1--C5   -0.027    0.021          0.877 0.012 0.111
#>  A2--C5    0.045    0.018          0.463 0.534 0.004
#>  A3--C5   -0.025    0.018          0.875 0.010 0.115
#>  A4--C5   -0.151    0.019          0.000 0.000 1.000
#>  A5--C5   -0.054    0.019          0.271 0.002 0.727
#>  C1--C5   -0.038    0.018          0.688 0.006 0.306
#>  C2--C5   -0.047    0.021          0.584 0.006 0.411
#>  C3--C5   -0.174    0.019          0.000 0.000 1.000
#>  C4--C5    0.357    0.020          0.000 1.000 0.000

# }
```
