# Prior Belief Gaussian Graphical Model

Incorporate prior information into the estimation of the conditional
dependence structure. This prior information is expressed as the prior
odds that each relation should be included in the graph.

## Usage

``` r
prior_belief_ggm(Y, prior_ggm, post_odds_cut = 3, ...)
```

## Arguments

- Y:

  Matrix (or data frame) of dimensions *n* (observations) by *p*
  (variables/nodes).

- prior_ggm:

  Matrix of dimensions *p* by *p*, encoding the prior odds for including
  each relation in the graph (see '`Details`')

- post_odds_cut:

  Numeric. Threshold for including an edge (defaults to 3). Note
  `post_odds` refers to posterior odds.

- ...:

  Additional arguments passed to
  [`explore`](https://rast-lab.github.io/BGGM/reference/explore.md).

## Value

An object including:

- **adj**: Adjacency matrix

- **post_prob**: Posterior probability for the alternative hypothesis.

## Details

Technically, the prior odds is not for including an edge in the graph,
but for (H1)/p(H0), where H1 captures the hypothesized edge size and H0
is the null model (see Williams2019_bf) . Accordingly, setting an entry
in `prior_ggm` to, say, 10, encodes a prior belief that H1 is 10 times
more likely than H0. Further, setting an entry in `prior_ggm` to 1
results in equal prior odds (the default in
[`select.explore`](https://rast-lab.github.io/BGGM/reference/select.explore.md)).

## Examples

``` r
# \donttest{
# Assume perfect prior information
# synthetic ggm
p <- 20
main <- gen_net()

# prior odds 10:1, assuming graph is known
prior_ggm <- ifelse(main$adj == 1, 10, 1)

# generate data
y <- MASS::mvrnorm(n = 200,
                   mu = rep(0, 20),
                   Sigma = main$cors)

# prior est
prior_est <- prior_belief_ggm(Y = y,
                              prior_ggm = prior_ggm,
                              progress = FALSE)

# check scores
BGGM:::performance(Estimate = prior_est$adj,
                   True = main$adj)
#> $results
#>       measure     score
#> 1 Specificity 0.9924812
#> 2 Sensitivity 0.6491228
#> 3   Precision 0.9736842
#> 4      Recall 0.6491228
#> 5    F1_score 0.7789474
#> 6         MCC 0.7350497
#> 

# default in BGGM
default_est <- select(explore(y, progress = FALSE))

# check scores
BGGM:::performance(Estimate = default_est$Adj_10,
                   True = main$adj)
#> $results
#>       measure     score
#> 1 Specificity 0.9849624
#> 2 Sensitivity 0.3333333
#> 3   Precision 0.9047619
#> 4      Recall 0.3333333
#> 5    F1_score 0.4871795
#> 6         MCC 0.4652015
#> 

# }
```
