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
#> B1 0.000  0.264 0.023  0.357 0.121
#> B2 0.264  0.000 0.543 -0.102 0.129
#> B3 0.023  0.543 0.000  0.266 0.175
#> B4 0.357 -0.102 0.266  0.000 0.365
#> B5 0.121  0.129 0.175  0.365 0.000
# }
```
