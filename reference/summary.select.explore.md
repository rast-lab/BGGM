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
#>  A1--A2   -0.244    0.020          0.000 0.000 1.000
#>  A1--A3   -0.106    0.021          0.000 0.000 1.000
#>  A2--A3    0.288    0.020          0.000 1.000 0.000
#>  A1--A4   -0.013    0.020          0.997 0.001 0.002
#>  A2--A4    0.161    0.019          0.000 1.000 0.000
#>  A3--A4    0.160    0.020          0.000 1.000 0.000
#>  A1--A5   -0.015    0.021          0.996 0.001 0.003
#>  A2--A5    0.144    0.019          0.000 1.000 0.000
#>  A3--A5    0.354    0.018          0.000 1.000 0.000
#>  A4--A5    0.114    0.019          0.000 1.000 0.000
#>  A1--C1    0.052    0.019          0.217 0.781 0.002
#>  A2--C1    0.004    0.020          0.998 0.001 0.001
#>  A3--C1    0.007    0.018          0.998 0.001 0.001
#>  A4--C1   -0.046    0.020          0.678 0.003 0.319
#>  A5--C1    0.062    0.020          0.043 0.956 0.001
#>  A1--C2    0.072    0.020          0.001 0.999 0.000
#>  A2--C2    0.006    0.021          0.998 0.001 0.001
#>  A3--C2    0.032    0.019          0.971 0.028 0.001
#>  A4--C2    0.152    0.019          0.000 1.000 0.000
#>  A5--C2   -0.028    0.021          0.986 0.001 0.012
#>  C1--C2    0.300    0.020          0.000 1.000 0.000
#>  A1--C3    0.040    0.019          0.833 0.165 0.002
#>  A2--C3    0.125    0.019          0.000 1.000 0.000
#>  A3--C3   -0.012    0.020          0.997 0.001 0.002
#>  A4--C3   -0.032    0.020          0.975 0.001 0.023
#>  A5--C3    0.019    0.021          0.995 0.004 0.001
#>  C1--C3    0.128    0.019          0.000 1.000 0.000
#>  C2--C3    0.184    0.019          0.000 1.000 0.000
#>  A1--C4    0.122    0.020          0.000 1.000 0.000
#>  A2--C4   -0.018    0.021          0.996 0.001 0.004
#>  A3--C4    0.013    0.020          0.997 0.002 0.001
#>  A4--C4    0.015    0.018          0.997 0.003 0.001
#>  A5--C4    0.004    0.019          0.998 0.001 0.001
#>  C1--C4   -0.154    0.019          0.000 0.000 1.000
#>  C2--C4   -0.188    0.020          0.000 0.000 1.000
#>  C3--C4   -0.122    0.020          0.000 0.000 1.000
#>  A1--C5   -0.025    0.020          0.990 0.001 0.009
#>  A2--C5    0.043    0.020          0.811 0.186 0.003
#>  A3--C5   -0.021    0.018          0.994 0.001 0.005
#>  A4--C5   -0.152    0.020          0.000 0.000 1.000
#>  A5--C5   -0.055    0.020          0.157 0.002 0.841
#>  C1--C5   -0.038    0.018          0.881 0.002 0.117
#>  C2--C5   -0.042    0.021          0.868 0.003 0.129
#>  C3--C5   -0.176    0.021          0.000 0.000 1.000
#>  C4--C5    0.359    0.020          0.000 1.000 0.000

# }
```
