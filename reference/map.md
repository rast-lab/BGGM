# Maximum A Posteriori Precision Matrix

Maximum A Posteriori Precision Matrix

## Usage

``` r
map(Y)
```

## Arguments

- Y:

  Matrix (or data frame) of dimensions *n* (observations) by *p*
  (variables).

## Value

An object of class `map`, including the precision matrix, partial
correlation matrix, and regression parameters.

## Examples

``` r
Y <- BGGM::bfi[, 1:5]

# map
map <- map(Y)
map
#> BGGM: Bayesian Gaussian Graphical Models 
#> --- 
#> Method: Maximum A Posteriori 
#> --- 
#>        [,1]   [,2]   [,3]   [,4]   [,5]
#> [1,]  0.000 -0.240 -0.107 -0.007 -0.009
#> [2,] -0.240  0.000  0.286  0.165  0.156
#> [3,] -0.107  0.286  0.000  0.178  0.359
#> [4,] -0.007  0.165  0.178  0.000  0.121
#> [5,] -0.009  0.156  0.359  0.121  0.000
```
