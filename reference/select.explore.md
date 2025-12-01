# Graph selection for `explore` Objects

Provides the selected graph based on the Bayes factor (Williams and
Mulder 2019) .

## Usage

``` r
# S3 method for class 'explore'
select(object, BF_cut = 3, alternative = "two.sided", ...)
```

## Arguments

- object:

  An object of class `explore.default`

- BF_cut:

  Numeric. Threshold for including an edge (defaults to 3).

- alternative:

  A character string specifying the alternative hypothesis. It must be
  one of "two.sided" (default), "greater", "less", or "exhaustive". See
  note for further details.

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
