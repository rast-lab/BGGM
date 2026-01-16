# Plot: Prior Distribution

Visualize the implied prior distribution for the partial correlations.
This is particularly useful for the Bayesian hypothesis testing methods.

## Usage

``` r
plot_prior(prior_sd = 0.5, iter = 5000)
```

## Arguments

- prior_sd:

  Scale of the prior distribution, approximately the standard deviation
  of a beta distribution (defaults to 0.5).

- iter:

  Number of iterations (prior samples; defaults to 5000).

## Value

A `ggplot` object.

## Examples

``` r
# note: iter = 250 for demonstrative purposes

plot_prior(prior_sd = 0.25, iter = 250)
```
