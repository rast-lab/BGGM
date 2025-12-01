# Graph Selection for `estimate` Objects

Provides the selected graph based on credible intervals for the partial
correlations that did not contain zero (Williams 2018) .

## Usage

``` r
# S3 method for class 'estimate'
select(object, cred = 0.95, alternative = "two.sided", ...)
```

## Arguments

- object:

  An object of class `estimate.default`.

- cred:

  Numeric. The credible interval width for selecting the graph (defaults
  to 0.95; must be between 0 and 1).

- alternative:

  A character string specifying the alternative hypothesis. It must be
  one of "two.sided" (default), "greater" or "less". See note for futher
  details.

- ...:

  Currently ignored.

## Value

The returned object of class `select.estimate` contains a lot of
information that is used for printing and plotting the results. For
users of **BGGM**, the following are the useful objects:

- `pcor_adj` Selected partial correlation matrix (weighted adjacency).

- `adj` Adjacency matrix for the selected edges

- `object` An object of class `estimate` (the fitted model).

## Details

This package was built for the social-behavioral sciences in particular.
In these applications, there is strong theory that expects *all* effects
to be positive. This is known as a "positive manifold" and this notion
has a rich tradition in psychometrics. Hence, this can be incorporated
into the graph with `alternative = "greater"`. This results in the
estimated structure including only positive edges.

## References

Williams DR (2018). “Bayesian Estimation for Gaussian Graphical Models:
Structure Learning, Predictability, and Network Comparisons.” *arXiv*.
[doi:10.31234/OSF.IO/X8DPR](https://doi.org/10.31234/OSF.IO/X8DPR) .

## See also

[`estimate`](https://rast-lab.github.io/BGGM/reference/estimate.md) and
[`ggm_compare_estimate`](https://rast-lab.github.io/BGGM/reference/ggm_compare_estimate.md)
for several examples.

## Examples

``` r
# \donttest{
# note: iter = 250 for demonstrative purposes

# data
Y <- bfi[,1:10]

# estimate
fit <- estimate(Y, iter = 250,
                progress = FALSE)


# select edge set
E <- select(fit)

# }
```
