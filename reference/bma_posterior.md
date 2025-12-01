# Compute Posterior Distributions from Graph Search Results

The \`bma_posterior\` function samples posterior distributions of graph
parameters (e.g., partial correlations or precision matrices) based on
the graph structures sampled during a Bayesian graph search performed by
[`ggm_search`](https://rast-lab.github.io/BGGM/reference/ggm_search.md).

## Usage

``` r
bma_posterior(object, param = "pcor", iter = 5000, progress = TRUE)
```

## Arguments

- object:

  A ggm_search object

- param:

  Compute BMA on either partial correlations "pcor" (default) or on
  precision matrix "Theta".

- iter:

  Number of samples to be drawn, defaults to 5,000

- progress:

  Show progress bar, defaults to TRUE

## Value

A list containing posterior samples and the Bayesian Model Averaged
parameter estimates.

## Details

This function incorporates uncertainty in both graph structure and
parameter estimation, providing Bayesian Model Averaged (BMA) parameter
estimates.

Use \`bma_posterior\` when detailed posterior inference on graph
parameters is needed, or to refine results obtained from \`ggm_search\`.

## See also

[`ggm_search`](https://rast-lab.github.io/BGGM/reference/ggm_search.md)
