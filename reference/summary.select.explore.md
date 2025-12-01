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
#>  A1--A2   -0.243    0.020          0.000 0.000 1.000
#>  A1--A3   -0.108    0.018          0.000 0.000 1.000
#>  A2--A3    0.289    0.019          0.000 1.000 0.000
#>  A1--A4   -0.015    0.019          0.997 0.001 0.003
#>  A2--A4    0.159    0.020          0.000 1.000 0.000
#>  A3--A4    0.157    0.020          0.000 1.000 0.000
#>  A1--A5   -0.016    0.020          0.996 0.001 0.003
#>  A2--A5    0.144    0.019          0.000 1.000 0.000
#>  A3--A5    0.353    0.019          0.000 1.000 0.000
#>  A4--A5    0.118    0.019          0.000 1.000 0.000
#>  A1--C1    0.054    0.019          0.164 0.833 0.002
#>  A2--C1    0.004    0.020          0.998 0.001 0.001
#>  A3--C1    0.007    0.019          0.998 0.001 0.001
#>  A4--C1   -0.049    0.021          0.646 0.003 0.351
#>  A5--C1    0.063    0.020          0.019 0.981 0.001
#>  A1--C2    0.072    0.019          0.000 1.000 0.000
#>  A2--C2    0.007    0.020          0.998 0.002 0.001
#>  A3--C2    0.034    0.020          0.961 0.037 0.002
#>  A4--C2    0.149    0.021          0.000 1.000 0.000
#>  A5--C2   -0.032    0.022          0.979 0.001 0.019
#>  C1--C2    0.299    0.019          0.000 1.000 0.000
#>  A1--C3    0.042    0.020          0.827 0.170 0.003
#>  A2--C3    0.125    0.019          0.000 1.000 0.000
#>  A3--C3   -0.013    0.020          0.997 0.001 0.002
#>  A4--C3   -0.030    0.018          0.977 0.001 0.022
#>  A5--C3    0.022    0.019          0.993 0.006 0.001
#>  C1--C3    0.126    0.018          0.000 1.000 0.000
#>  C2--C3    0.183    0.020          0.000 1.000 0.000
#>  A1--C4    0.123    0.020          0.000 1.000 0.000
#>  A2--C4   -0.016    0.020          0.996 0.001 0.003
#>  A3--C4    0.014    0.018          0.997 0.002 0.001
#>  A4--C4    0.012    0.021          0.997 0.002 0.001
#>  A5--C4    0.005    0.021          0.998 0.001 0.001
#>  C1--C4   -0.158    0.018          0.000 0.000 1.000
#>  C2--C4   -0.187    0.021          0.000 0.000 1.000
#>  C3--C4   -0.123    0.019          0.000 0.000 1.000
#>  A1--C5   -0.023    0.021          0.992 0.001 0.007
#>  A2--C5    0.045    0.019          0.660 0.337 0.003
#>  A3--C5   -0.024    0.020          0.992 0.001 0.007
#>  A4--C5   -0.149    0.019          0.000 0.000 1.000
#>  A5--C5   -0.054    0.022          0.486 0.004 0.511
#>  C1--C5   -0.037    0.019          0.912 0.002 0.086
#>  C2--C5   -0.045    0.019          0.655 0.003 0.342
#>  C3--C5   -0.176    0.019          0.000 0.000 1.000
#>  C4--C5    0.358    0.021          0.000 1.000 0.000

# }
```
