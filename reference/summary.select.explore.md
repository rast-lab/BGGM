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
#>  A1--A2   -0.244    0.018          0.000 0.000 1.000
#>  A1--A3   -0.107    0.019          0.000 0.000 1.000
#>  A2--A3    0.287    0.019          0.000 1.000 0.000
#>  A1--A4   -0.016    0.020          0.996 0.001 0.003
#>  A2--A4    0.162    0.020          0.000 1.000 0.000
#>  A3--A4    0.160    0.020          0.000 1.000 0.000
#>  A1--A5   -0.015    0.018          0.996 0.001 0.003
#>  A2--A5    0.143    0.021          0.000 1.000 0.000
#>  A3--A5    0.356    0.021          0.000 1.000 0.000
#>  A4--A5    0.113    0.020          0.000 1.000 0.000
#>  A1--C1    0.050    0.019          0.265 0.732 0.003
#>  A2--C1    0.002    0.019          0.998 0.001 0.001
#>  A3--C1    0.010    0.019          0.997 0.002 0.001
#>  A4--C1   -0.047    0.019          0.577 0.003 0.420
#>  A5--C1    0.061    0.020          0.026 0.973 0.001
#>  A1--C2    0.071    0.020          0.002 0.998 0.000
#>  A2--C2    0.007    0.019          0.998 0.002 0.001
#>  A3--C2    0.030    0.019          0.974 0.025 0.001
#>  A4--C2    0.151    0.019          0.000 1.000 0.000
#>  A5--C2   -0.028    0.020          0.984 0.001 0.015
#>  C1--C2    0.301    0.022          0.000 1.000 0.000
#>  A1--C3    0.042    0.020          0.827 0.170 0.003
#>  A2--C3    0.125    0.021          0.000 1.000 0.000
#>  A3--C3   -0.013    0.019          0.997 0.001 0.002
#>  A4--C3   -0.034    0.020          0.961 0.002 0.037
#>  A5--C3    0.021    0.021          0.993 0.006 0.001
#>  C1--C3    0.126    0.020          0.000 1.000 0.000
#>  C2--C3    0.182    0.019          0.000 1.000 0.000
#>  A1--C4    0.120    0.019          0.000 1.000 0.000
#>  A2--C4   -0.022    0.020          0.993 0.001 0.006
#>  A3--C4    0.013    0.018          0.997 0.002 0.001
#>  A4--C4    0.018    0.018          0.995 0.004 0.001
#>  A5--C4    0.007    0.018          0.998 0.001 0.001
#>  C1--C4   -0.155    0.021          0.000 0.000 1.000
#>  C2--C4   -0.188    0.022          0.000 0.000 1.000
#>  C3--C4   -0.124    0.020          0.000 0.000 1.000
#>  A1--C5   -0.026    0.018          0.986 0.001 0.013
#>  A2--C5    0.044    0.020          0.766 0.230 0.003
#>  A3--C5   -0.024    0.020          0.991 0.001 0.008
#>  A4--C5   -0.152    0.017          0.000 0.000 1.000
#>  A5--C5   -0.056    0.021          0.189 0.003 0.808
#>  C1--C5   -0.041    0.020          0.880 0.003 0.117
#>  C2--C5   -0.043    0.020          0.828 0.003 0.169
#>  C3--C5   -0.177    0.022          0.000 0.000 1.000
#>  C4--C5    0.357    0.021          0.000 1.000 0.000

# }
```
