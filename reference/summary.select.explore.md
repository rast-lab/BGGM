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
#>  A1--A2   -0.243    0.018          0.000 0.000 1.000
#>  A1--A3   -0.105    0.022          0.000 0.000 1.000
#>  A2--A3    0.286    0.020          0.000 1.000 0.000
#>  A1--A4   -0.015    0.018          0.997 0.001 0.002
#>  A2--A4    0.159    0.020          0.000 1.000 0.000
#>  A3--A4    0.159    0.019          0.000 1.000 0.000
#>  A1--A5   -0.015    0.020          0.997 0.001 0.003
#>  A2--A5    0.147    0.019          0.000 1.000 0.000
#>  A3--A5    0.354    0.019          0.000 1.000 0.000
#>  A4--A5    0.116    0.020          0.000 1.000 0.000
#>  A1--C1    0.055    0.020          0.207 0.791 0.002
#>  A2--C1    0.002    0.021          0.998 0.001 0.001
#>  A3--C1    0.009    0.019          0.998 0.001 0.001
#>  A4--C1   -0.047    0.019          0.615 0.003 0.382
#>  A5--C1    0.061    0.020          0.030 0.969 0.001
#>  A1--C2    0.070    0.018          0.000 1.000 0.000
#>  A2--C2    0.008    0.020          0.998 0.001 0.001
#>  A3--C2    0.031    0.019          0.975 0.024 0.001
#>  A4--C2    0.150    0.021          0.000 1.000 0.000
#>  A5--C2   -0.029    0.019          0.984 0.001 0.015
#>  C1--C2    0.300    0.020          0.000 1.000 0.000
#>  A1--C3    0.042    0.019          0.828 0.169 0.002
#>  A2--C3    0.125    0.020          0.000 1.000 0.000
#>  A3--C3   -0.013    0.019          0.997 0.001 0.002
#>  A4--C3   -0.030    0.021          0.985 0.001 0.014
#>  A5--C3    0.019    0.020          0.995 0.004 0.001
#>  C1--C3    0.126    0.019          0.000 1.000 0.000
#>  C2--C3    0.183    0.018          0.000 1.000 0.000
#>  A1--C4    0.124    0.018          0.000 1.000 0.000
#>  A2--C4   -0.016    0.018          0.997 0.001 0.003
#>  A3--C4    0.011    0.021          0.997 0.002 0.001
#>  A4--C4    0.016    0.018          0.997 0.003 0.001
#>  A5--C4    0.006    0.019          0.998 0.001 0.001
#>  C1--C4   -0.159    0.020          0.000 0.000 1.000
#>  C2--C4   -0.187    0.019          0.000 0.000 1.000
#>  C3--C4   -0.125    0.019          0.000 0.000 1.000
#>  A1--C5   -0.025    0.020          0.991 0.001 0.008
#>  A2--C5    0.044    0.020          0.812 0.186 0.003
#>  A3--C5   -0.025    0.019          0.991 0.001 0.008
#>  A4--C5   -0.150    0.019          0.000 0.000 1.000
#>  A5--C5   -0.054    0.019          0.183 0.002 0.814
#>  C1--C5   -0.038    0.020          0.934 0.002 0.064
#>  C2--C5   -0.046    0.019          0.576 0.003 0.421
#>  C3--C5   -0.175    0.020          0.000 0.000 1.000
#>  C4--C5    0.358    0.020          0.000 1.000 0.000

# }
```
