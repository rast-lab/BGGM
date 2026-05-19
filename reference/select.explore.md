# Graph selection for `explore` Objects

Provides the selected graph based on the Bayes factor (Williams and
Mulder 2019) .

## Usage

``` r
# S3 method for class 'explore'
select(
  object,
  method = c("BF_cut", "BMA"),
  BF_cut = 3,
  prior.prob.H0 = 0.5,
  alternative = "two.sided",
  ...
)
```

## Arguments

- object:

  An object of class `explore.default`

- method:

  Character string specifying the edge selection method. Options
  include:

  - `"BF_cut"`: Select edges based on a Bayes factor threshold. This is
    the original approach described in (Williams and Mulder 2019) .

  - `"BMA"`: Bayesian model averaging based on posterior model
    probabilities. For each edge, posterior draws are generated from a
    mixture distribution placing mass at zero under the null model and
    using posterior draws from the alternative model otherwise. Reported
    edges are based on the posterior median of these draws.

- BF_cut:

  Numeric. Bayes factor threshold for including an edge when
  `method = "BF_cut"` (defaults to 3).

- prior.prob.H0:

  Numeric between 0 and 1. Prior probability assigned to the null
  hypothesis for each edge when `method = "BMA"` (defaults to `0.5`).

- alternative:

  A character string specifying the alternative hypothesis. It must be
  one of "two.sided" (default), "greater", "less", or "exhaustive". See
  note for further details. Note that `alternative = "exhaustive"` is
  not supported for `method = "BMA"`.

- ...:

  Currently ignored.

## Value

The returned object of class `select.explore` contains a lot of
information that is used for printing and plotting the results. For
users of **BGGM**, the following are the useful objects:

`alternative = "two.sided"`

- `pcor_mat_zero` Selected partial correlation matrix (weighted
  adjacency).

- `pcor_mat` Partial correlation matrix (posterior mean).

- `Adj_10` Adjacency matrix for the selected edges.

- `Adj_01` Adjacency matrix for which there was evidence for the null
  hypothesis.

`alternative = "greater"` and `"less"`

- `pcor_mat_zero` Selected partial correlation matrix (weighted
  adjacency).

- `pcor_mat` Partial correlation matrix (posterior mean).

- `Adj_20` Adjacency matrix for the selected edges.

- `Adj_02` Adjacency matrix for which there was evidence for the null
  hypothesis (see note).

`alternative = "exhaustive"`

- `post_prob` A data frame that included the posterior hypothesis
  probabilities.

- `neg_mat` Adjacency matrix for which there was evidence for negative
  edges.

- `pos_mat` Adjacency matrix for which there was evidence for positive
  edges.

- `neg_mat` Adjacency matrix for which there was evidence for the null
  hypothesis (see note).

- `pcor_mat` Partial correlation matrix (posterior mean). The weighted
  adjacency matrices can be computed by multiplying `pcor_mat` with an
  adjacency matrix.

## Details

Exhaustive provides the posterior hypothesis probabilities for a
positive, negative, or null relation (see Table 3 in Williams and Mulder
2019) .

`method = "BF_cut"` performs edge selection using Bayes factor
thresholding.

`method = "BMA"` performs Bayesian model averaging by generating
posterior draws from a spike-and-slab style mixture distribution for
each edge. The spike corresponds to the null hypothesis (exactly zero
partial correlation), whereas the slab corresponds to posterior draws
under the alternative hypothesis. Posterior model probabilities are
computed from the Bayes factors and `prior.prob.H0`. The selected
network is based on the posterior median of the resulting draws.

## Note

Care must be taken with the options `alternative = "less"` and
`alternative = "greater"`. This is because the full parameter space is
not included, such, for `alternative = "greater"`, there can be evidence
for the "null" when the relation is negative. This inference is correct:
the null model better predicted the data than the positive model. But
note this is relative and does **not** provide absolute evidence for the
null hypothesis.

## References

Williams DR, Mulder J (2019). “Bayesian Hypothesis Testing for Gaussian
Graphical Models: Conditional Independence and Order Constraints.”
*PsyArXiv*.
[doi:10.31234/osf.io/ypxd8](https://doi.org/10.31234/osf.io/ypxd8) .

## See also

[`explore`](https://rast-lab.github.io/BGGM/reference/explore.md) and
[`ggm_compare_explore`](https://rast-lab.github.io/BGGM/reference/ggm_compare_explore.md)
for several examples.

## Examples

``` r

# \donttest{
#################
### example 1 ###
#################

#  data
Y <- bfi[,1:10]

# fit model
fit <- explore(Y, progress = FALSE)

# edge set
E <- select(fit,
            alternative = "exhaustive")

# }
```
