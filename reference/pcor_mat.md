# Extract the Partial Correlation Matrix

Extract the partial correlation matrix (posterior mean) from
[`estimate`](https://rast-lab.github.io/BGGM/reference/estimate.md),
[`explore`](https://rast-lab.github.io/BGGM/reference/explore.md),
[`ggm_compare_estimate`](https://rast-lab.github.io/BGGM/reference/ggm_compare_estimate.md),
and
[`ggm_compare_explore`](https://rast-lab.github.io/BGGM/reference/ggm_compare_explore.md)
objects. It is also possible to extract the partial correlation
differences for
[`ggm_compare_estimate`](https://rast-lab.github.io/BGGM/reference/ggm_compare_estimate.md)
and
[`ggm_compare_explore`](https://rast-lab.github.io/BGGM/reference/ggm_compare_explore.md)
objects.

## Usage

``` r
pcor_mat(object, difference = FALSE, ...)
```

## Arguments

- object:

  A model estimated with **BGGM**. All classes are supported, assuming
  there is matrix to be extracted.

- difference:

  Logical. Should the difference be returned (defaults to `FALSE`) ?
  Note that this assumes there is a difference (e.g., an object of class
  `ggm_compare_estimate`) and ignored otherwise.

- ...:

  Currently ignored.

## Value

The estimated partial correlation matrix.

## Examples

``` r
# \donttest{
# note: iter = 250 for demonstrative purposes

# data
Y <- ptsd[,1:5] + 1

# ordinal
fit <- estimate(Y, type = "ordinal",
                iter = 250,
                progress = FALSE)

pcor_mat(fit)
#>       B1     B2    B3     B4    B5
#> B1 0.000  0.231 0.034  0.344 0.143
#> B2 0.231  0.000 0.534 -0.085 0.114
#> B3 0.034  0.534 0.000  0.265 0.188
#> B4 0.344 -0.085 0.265  0.000 0.366
#> B5 0.143  0.114 0.188  0.366 0.000
# }
```
