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
#>  A1--A2   -0.246    0.019          0.000 0.000 1.000
#>  A1--A3   -0.107    0.022          0.000 0.000 1.000
#>  A2--A3    0.288    0.020          0.000 1.000 0.000
#>  A1--A4   -0.013    0.020          0.997 0.001 0.002
#>  A2--A4    0.161    0.019          0.000 1.000 0.000
#>  A3--A4    0.158    0.020          0.000 1.000 0.000
#>  A1--A5   -0.016    0.020          0.996 0.001 0.003
#>  A2--A5    0.145    0.019          0.000 1.000 0.000
#>  A3--A5    0.355    0.021          0.000 1.000 0.000
#>  A4--A5    0.113    0.021          0.000 1.000 0.000
#>  A1--C1    0.050    0.020          0.424 0.573 0.003
#>  A2--C1    0.002    0.019          0.998 0.001 0.001
#>  A3--C1    0.008    0.021          0.997 0.002 0.001
#>  A4--C1   -0.046    0.020          0.700 0.003 0.297
#>  A5--C1    0.063    0.020          0.018 0.981 0.001
#>  A1--C2    0.072    0.020          0.001 0.999 0.000
#>  A2--C2    0.008    0.021          0.997 0.002 0.001
#>  A3--C2    0.034    0.019          0.960 0.039 0.002
#>  A4--C2    0.151    0.019          0.000 1.000 0.000
#>  A5--C2   -0.030    0.018          0.973 0.001 0.025
#>  C1--C2    0.301    0.020          0.000 1.000 0.000
#>  A1--C3    0.042    0.020          0.855 0.142 0.003
#>  A2--C3    0.125    0.021          0.000 1.000 0.000
#>  A3--C3   -0.012    0.020          0.997 0.001 0.002
#>  A4--C3   -0.033    0.019          0.961 0.002 0.037
#>  A5--C3    0.021    0.018          0.993 0.006 0.001
#>  C1--C3    0.127    0.019          0.000 1.000 0.000
#>  C2--C3    0.181    0.017          0.000 1.000 0.000
#>  A1--C4    0.121    0.018          0.000 1.000 0.000
#>  A2--C4   -0.019    0.020          0.995 0.001 0.004
#>  A3--C4    0.011    0.020          0.997 0.002 0.001
#>  A4--C4    0.019    0.019          0.995 0.005 0.001
#>  A5--C4    0.005    0.019          0.998 0.001 0.001
#>  C1--C4   -0.155    0.018          0.000 0.000 1.000
#>  C2--C4   -0.189    0.019          0.000 0.000 1.000
#>  C3--C4   -0.124    0.021          0.000 0.000 1.000
#>  A1--C5   -0.024    0.020          0.991 0.001 0.008
#>  A2--C5    0.043    0.019          0.723 0.274 0.003
#>  A3--C5   -0.022    0.019          0.993 0.001 0.006
#>  A4--C5   -0.151    0.019          0.000 0.000 1.000
#>  A5--C5   -0.054    0.020          0.258 0.003 0.740
#>  C1--C5   -0.037    0.017          0.852 0.002 0.146
#>  C2--C5   -0.044    0.019          0.711 0.003 0.286
#>  C3--C5   -0.174    0.021          0.000 0.000 1.000
#>  C4--C5    0.360    0.020          0.000 1.000 0.000

# }
```
