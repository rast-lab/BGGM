# Perform Bayesian Graph Search and Optional Model Averaging

The \`ggm_search\` function performs a Bayesian graph search to identify
the most probable graph structure (MAP solution) using the
Metropolis-Hastings algorithm. It also computes an optional Bayesian
Model Averaged (BMA) solution across the graph structures sampled during
the search.

## Usage

``` r
ggm_search(
  x,
  n = NULL,
  method = "mc3",
  prior_prob = 0.3,
  iter = 5000,
  stop_early = 1000,
  bma_mean = TRUE,
  seed = NULL,
  progress = TRUE,
  ...
)
```

## Arguments

- x:

  Data, either raw data or covariance matrix

- n:

  For x = covariance matrix, provide number of observations

- method:

  mc3 defaults to MH sampling

- prior_prob:

  Prior prbability of sparseness.

- iter:

  Number of iterations \#@param burn_in Burn in. Defaults to iter/2

- stop_early:

  Default to 1000. Stop MH algorithm if proposals keep being rejected
  (stopping by default after 1000 rejections).

- bma_mean:

  Compute Bayesian Model Averaged solution

- seed:

  Set seed. Current default is to set R's random seed.

- progress:

  Show progress bar, defaults to TRUE

- ...:

  Not currently in use

## Value

A list containing the MAP graph structure, BMA solution (if specified),
and posterior probabilities of the sampled graphs.

## Details

This function is ideal for exploring the graph space and obtaining an
initial estimate of the graph structure or adjacency matrix.

To refine the results or compute posterior distributions of graph
parameters (e.g., partial correlations), use the
[`bma_posterior`](https://rast-lab.github.io/BGGM/reference/bma_posterior.md)
function, which builds on the output of \`ggm_search\` to account for
parameter uncertainty.

## See also

[`bma_posterior`](https://rast-lab.github.io/BGGM/reference/bma_posterior.md)

## Author

Donny Williams and Philippe Rast
