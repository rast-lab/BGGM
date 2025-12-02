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
#>  A1--A3   -0.106    0.020          0.000 0.000 1.000
#>  A2--A3    0.285    0.019          0.000 1.000 0.000
#>  A1--A4   -0.014    0.018          0.997 0.001 0.003
#>  A2--A4    0.163    0.020          0.000 1.000 0.000
#>  A3--A4    0.160    0.020          0.000 1.000 0.000
#>  A1--A5   -0.017    0.018          0.996 0.001 0.003
#>  A2--A5    0.146    0.020          0.000 1.000 0.000
#>  A3--A5    0.355    0.021          0.000 1.000 0.000
#>  A4--A5    0.113    0.020          0.000 1.000 0.000
#>  A1--C1    0.051    0.020          0.378 0.619 0.003
#>  A2--C1    0.004    0.019          0.998 0.001 0.001
#>  A3--C1    0.008    0.020          0.997 0.002 0.001
#>  A4--C1   -0.048    0.020          0.601 0.003 0.396
#>  A5--C1    0.063    0.019          0.008 0.992 0.000
#>  A1--C2    0.072    0.019          0.000 1.000 0.000
#>  A2--C2    0.007    0.019          0.998 0.002 0.001
#>  A3--C2    0.032    0.020          0.970 0.028 0.002
#>  A4--C2    0.150    0.018          0.000 1.000 0.000
#>  A5--C2   -0.031    0.019          0.972 0.001 0.026
#>  C1--C2    0.299    0.019          0.000 1.000 0.000
#>  A1--C3    0.041    0.021          0.901 0.096 0.003
#>  A2--C3    0.125    0.019          0.000 1.000 0.000
#>  A3--C3   -0.012    0.020          0.997 0.001 0.002
#>  A4--C3   -0.033    0.018          0.953 0.002 0.045
#>  A5--C3    0.021    0.019          0.993 0.006 0.001
#>  C1--C3    0.126    0.019          0.000 1.000 0.000
#>  C2--C3    0.181    0.019          0.000 1.000 0.000
#>  A1--C4    0.124    0.019          0.000 1.000 0.000
#>  A2--C4   -0.018    0.019          0.995 0.001 0.004
#>  A3--C4    0.013    0.018          0.997 0.002 0.001
#>  A4--C4    0.017    0.020          0.995 0.004 0.001
#>  A5--C4    0.005    0.019          0.998 0.001 0.001
#>  C1--C4   -0.156    0.021          0.000 0.000 1.000
#>  C2--C4   -0.191    0.018          0.000 0.000 1.000
#>  C3--C4   -0.126    0.019          0.000 0.000 1.000
#>  A1--C5   -0.026    0.018          0.985 0.001 0.014
#>  A2--C5    0.044    0.019          0.693 0.304 0.003
#>  A3--C5   -0.022    0.020          0.992 0.001 0.007
#>  A4--C5   -0.152    0.019          0.000 0.000 1.000
#>  A5--C5   -0.056    0.020          0.165 0.002 0.833
#>  C1--C5   -0.038    0.019          0.906 0.002 0.092
#>  C2--C5   -0.045    0.019          0.586 0.003 0.411
#>  C3--C5   -0.177    0.020          0.000 0.000 1.000
#>  C4--C5    0.356    0.020          0.000 1.000 0.000

# }
```
