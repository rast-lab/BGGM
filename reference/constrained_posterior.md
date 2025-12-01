# Constrained Posterior Distribution

Compute the posterior distribution with off-diagonal elements of the
precision matrix constrained to zero.

## Usage

``` r
constrained_posterior(
  object,
  adj,
  method = "direct",
  iter = 5000,
  progress = TRUE,
  ...
)
```

## Arguments

- object:

  An object of class `estimate` or `explore`

- adj:

  A `p` by `p` adjacency matrix. The zero entries denote the elements
  that should be constrained to zero.

- method:

  Character string. Which method should be used ? Defaults to the
  "direct sampler" (i.e., `method = "direct"`) described in page 122,
  section 2.4, Lenkoski (2013) . The other option is a
  Metropolis-Hastings algorithm (`MH`). See details.

- iter:

  Number of iterations (posterior samples; defaults to 5000).

- progress:

  Logical. Should a progress bar be included (defaults to `TRUE`) ?

- ...:

  Currently ignored.

## Value

An object of class `contrained`, including

- `precision_mean` The posterior mean for the precision matrix.

- `pcor_mean` The posterior mean for the precision matrix.

- `precision_samps` A 3d array of dimension `p` by `p` by `iter`
  including the sampled precision matrices.

- `pcor_samps` A 3d array of dimension `p` by `p` by `iter` including
  sampled partial correlations matrices.

## References

Lenkoski A (2013). “A direct sampler for G-Wishart variates.” *Stat*,
**2**(1), 119–128.

## Examples

``` r
# \donttest{

# data
Y <- bfi[,1:10]

# sample posterior
fit <- estimate(Y, iter = 100)
#> BGGM: Posterior Sampling 
#> BGGM: Finished

# select graph
sel <- select(fit)

# constrained posterior
post <- constrained_posterior(object = fit,
                              adj = sel$adj,
                              iter = 100,
                              progress = FALSE)

# }
```
