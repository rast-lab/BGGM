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
    probabilities. For each edge, the posterior is a mixture
    distribution placing mass at zero under the null model and using the
    posterior under the alternative model otherwise. Reported edges are
    based on the median of this mixture.

- BF_cut:

  Numeric. Evidence threshold for including an edge when
  `method = "BF_cut"` (defaults to 3). An edge is selected when the
  posterior probability of the hypothesis exceeds
  `BF_cut / (BF_cut + 1)`. With the default `prior.prob.H0 = 0.5` this
  is equivalent to a Bayes factor of `BF_cut` against the competing
  hypothesis; for other values of `prior.prob.H0` it is a cutoff on the
  posterior probability.

- prior.prob.H0:

  Numeric between 0 and 1. Prior probability assigned to the null
  hypothesis for each edge (defaults to `0.5`). It is used for the
  posterior hypothesis probabilities and the edge inclusion
  probabilities under both `method = "BF_cut"` and `method = "BMA"`. For
  `alternative = "exhaustive"` the remaining `1 - prior.prob.H0` is
  split equally over the positive and the negative hypothesis, so that
  splitting the alternative by sign leaves \\P(H_0 \mid Y)\\ unchanged.

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

- `incl_prob` Matrix of posterior edge inclusion probabilities, \\P(H_1
  \mid Y)\\, based on `BF_10` and prior inclusion probability
  `1 - prior.prob.H0`.

`alternative = "greater"` and `"less"`

- `pcor_mat_zero` Selected partial correlation matrix (weighted
  adjacency).

- `pcor_mat` Partial correlation matrix (posterior mean).

- `Adj_20` Adjacency matrix for the selected edges.

- `Adj_02` Adjacency matrix for which there was evidence for the null
  hypothesis (see note).

- `incl_prob` Matrix of posterior probabilities of the one-sided
  hypothesis against the null, based on `BF_20` and prior probability
  `1 - prior.prob.H0`.

`alternative = "exhaustive"`

- `post_prob` A data frame of the posterior hypothesis probabilities
  \\P(H_0 \mid Y)\\, \\P(H\_+ \mid Y)\\, and \\P(H\_- \mid Y)\\ for each
  relation (a null, positive, or negative partial correlation).

  For `method = "BF_cut"` the following are hard hypothesis assignments;
  for `method = "BMA"` they classify the sign of the model-averaged
  posterior median (`pcor_mat_zero`), not the most probable hypothesis:

- `pos_mat` Adjacency matrix for positive edges.

- `neg_mat` Adjacency matrix for negative edges.

- `null_mat` Adjacency matrix for null edges (see note).

- `incl_prob` Matrix of posterior edge inclusion probabilities, \\1 -
  P(H_0 \mid Y)\\.

- `pcor_mat` Partial correlation matrix (posterior mean). The weighted
  adjacency matrices can be computed by multiplying `pcor_mat` with an
  adjacency matrix.

- `pcor_mat_zero` Selected partial correlation matrix (weighted
  adjacency). For `method = "BF_cut"` this is the posterior mean of the
  selected edges and zero elsewhere; for `method = "BMA"` it is the
  model-averaged matrix, i.e. for each edge the posterior median of the
  three-state mixture over the null, positive, and negative hypotheses.

## Details

Exhaustive provides the posterior hypothesis probabilities for a
positive, negative, or null relation (see Table 3 in Williams and Mulder
2019) .

`method = "BF_cut"` selects an edge when its posterior inclusion
probability exceeds `BF_cut / (BF_cut + 1)` (0.75 for `BF_cut = 3`), and
calls an edge null when the posterior probability of the null hypothesis
exceeds that same cutoff. With the default `prior.prob.H0 = 0.5` this is
the Bayes factor threshold of (Williams and Mulder 2019) : `BF_cut = 3`
selects the edges with a Bayes factor larger than 3. For
`alternative = "exhaustive"` the inclusion probability is \\1 - P(H_0
\mid Y) = P(H\_+ \mid Y) + P(H\_- \mid Y)\\, which equals the inclusion
probability of `alternative = "two.sided"`, so both give the same
selected edges; a selected edge is labelled positive or negative
according to the larger of the two directional probabilities, which are
reported in addition. An edge can be assigned to none of the three
hypotheses.

`method = "BMA"` does not use `BF_cut`: an edge is selected when the
median of the model-averaged mixture is nonzero, which corresponds to an
inclusion probability above 0.5, so it selects more edges than
`method = "BF_cut"` with the default `BF_cut = 3`.

`method = "BMA"` performs Bayesian model averaging using a
spike-and-slab style mixture distribution for each edge. The spike
corresponds to the null hypothesis (exactly zero partial correlation),
whereas the slab corresponds to the posterior under the alternative
hypothesis, approximated by a normal distribution for the Fisher-z
transformed partial correlation (truncated to the positive or negative
half-line for one-sided hypotheses). Posterior model probabilities are
computed from the Bayes factors and `prior.prob.H0`. The selected
network is based on the median of this mixture, which is computed
exactly (no simulation), so the result is deterministic. For
`alternative = "exhaustive"` the mixture has three states – a spike at
zero (\\H_0\\), a positive slab (\\H\_+\\), and a negative slab
(\\H\_-\\) – mixed by the posterior hypothesis probabilities. The
model-averaged partial correlations are returned in `pcor_mat_zero`, and
`pos_mat`/`neg_mat`/`null_mat` classify each edge by the sign of that
model-averaged median.

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

# edge set (Bayes factor threshold)
E <- select(fit,
            alternative = "exhaustive")

# edge set (Bayesian model averaging), with prior P(H0) = 0.5
E <- select(fit,
            method = "BMA",
            alternative = "exhaustive")

# }
```
