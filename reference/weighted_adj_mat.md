# Extract the Weighted Adjacency Matrix

Extract the weighted adjacency matrix (posterior mean) from
[`estimate`](https://rast-lab.github.io/BGGM/reference/estimate.md),
[`explore`](https://rast-lab.github.io/BGGM/reference/explore.md),
[`ggm_compare_estimate`](https://rast-lab.github.io/BGGM/reference/ggm_compare_estimate.md),
and
[`ggm_compare_explore`](https://rast-lab.github.io/BGGM/reference/ggm_compare_explore.md)
objects.

## Usage

``` r
weighted_adj_mat(object, ...)
```

## Arguments

- object:

  A model estimated with **BGGM**. All classes are supported, assuming
  there is matrix to be extracted.

- ...:

  Currently ignored.

## Value

The weighted adjacency matrix (partial correlation matrix with zeros).

## Examples

``` r
# \donttest{
# note: iter = 250 for demonstrative purposes
Y <- bfi[,1:5]

# estimate
fit <- estimate(Y, iter = 250,
                progress = FALSE)

# select graph
E <- select(fit)

# extract weighted adj matrix
weighted_adj_mat(E)
#>        [,1]   [,2]   [,3]  [,4]  [,5]
#> [1,]  0.000 -0.241 -0.109 0.000 0.000
#> [2,] -0.241  0.000  0.287 0.165 0.156
#> [3,] -0.109  0.287  0.000 0.177 0.358
#> [4,]  0.000  0.165  0.177 0.000 0.122
#> [5,]  0.000  0.156  0.358 0.122 0.000

# }
```
