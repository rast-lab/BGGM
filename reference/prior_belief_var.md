# Prior Belief Graphical VAR

Prior Belief Graphical VAR

## Usage

``` r
prior_belief_var(
  Y,
  prior_temporal = NULL,
  post_odds_cut = 3,
  est_ggm = TRUE,
  prior_ggm = NULL,
  progress = TRUE,
  ...
)
```

## Arguments

- Y:

  Matrix (or data frame) of dimensions *n* (observations) by *p*
  (variables/nodes).

- prior_temporal:

  Matrix of dimensions *p* by *p*, encoding the prior odds for including
  each relation in the temporal graph (see '`Details`'). If null a
  matrix of 1's is used, resulting in equal prior odds.

- post_odds_cut:

  Numeric. Threshold for including an edge (defaults to 3). Note
  `post_odds` refers to posterior odds.

- est_ggm:

  Logical. Should the contemporaneous network be estimated (defaults to
  `TRUE`)?

- prior_ggm:

  Matrix of dimensions *p* by *p*, encoding the prior odds for including
  each relation in the graph (see '`Details`'). If null a matrix of 1's
  is used, resulting in equal prior odds.

- progress:

  Logical. Should a progress bar be included (defaults to `TRUE`) ?

- ...:

  Additional arguments passed to
  [`explore`](https://rast-lab.github.io/BGGM/reference/explore.md).
  Ignored if `prior_ggm = FALSE`.

## Value

An object including (`est_ggm = FALSE`):

- **adj**: Adjacency matrix

- **post_prob**: Posterior probability for the alternative hypothesis.

An object including (`est_ggm = TRUE`):

- **adj_temporal**: Adjacency matrix for the temporal network.

- **post_prob_temporal**: Posterior probability for the alternative
  hypothesis (temporal edge)

- **adj_ggm**: Adjacency matrix for the contemporaneous network (ggm).

- post_prob_ggm: Posterior probability for the alternative hypothesis
  (contemporaneous edge)

## Details

Technically, the prior odds is not for including an edge in the graph,
but for (H1)/p(H0), where H1 captures the hypothesized edge size and H0
is the null model (see Williams2019_bf) . Accordingly, setting an entry
in `prior_ggm` to, say, 10, encodes a prior belief that H1 is 10 times
more likely than H0. Further, setting an entry in `prior_ggm` or
`prior_var` to 1 results in equal prior odds (the default in
[`select.explore`](https://rast-lab.github.io/BGGM/reference/select.explore.md)).

## Note

The returned matrices are formatted with the rows indicating the outcome
and the columns the predictor. Hence, adj_temporal\[1,4\] is the
temporal relation of node 4 predicting node 1. This follows the
convention of the **vars** package (i.e., `Acoef`).

Further, in order to compute the Bayes factor the data is standardized
(mean = 0 and standard deviation = 1).

## Examples

``` r
# \donttest{
# affect data from 1 person
# (real data)
y <- na.omit(subset(ifit, id == 1)[,2:7])
p <- ncol(y)

# random prior graph
# (dont do this in practice!!)
prior_var = matrix(sample(c(1,10),
                   size = p^2, replace = TRUE),
                   nrow = p, ncol = p)

# fit model
fit <- prior_belief_var(y,
                        prior_temporal = prior_var,
                        post_odds_cut = 3)
#> testing temporal relations
#>   |                                                                              |                                                                      |   0%  |                                                                              |============                                                          |  17%  |                                                                              |=======================                                               |  33%  |                                                                              |===================================                                   |  50%  |                                                                              |===============================================                       |  67%  |                                                                              |==========================================================            |  83%  |                                                                              |======================================================================| 100%
#> 
#> 
#> testing contemporanenous relations
#> BGGM: Posterior Sampling 
#> BGGM: Prior Sampling 
#> BGGM: Finished
# }
```
