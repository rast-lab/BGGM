# Extract Posterior Samples

Extract posterior samples for all parameters.

## Usage

``` r
posterior_samples(object, ...)
```

## Arguments

- object:

  an object of class `estimate` or `explore`.

- ...:

  currently ignored.

## Value

A matrix of posterior samples for the partial correlation. Note that if
controlling for variables (e.g., formula `~ age`), the matrix also
includes the coefficients from each multivariate regression.

## Examples

``` r
# \donttest{
# note: iter = 250 for demonstrative purposes

########################################
### example 1: control  with formula ###
########################################
# (the following works with all data types)

# controlling for gender
Y <- bfi

# to control for only gender
# (remove education)
Y <- subset(Y, select = - education)

# fit model
fit <- estimate(Y, formula = ~ gender,
                iter = 250)
#> BGGM: Posterior Sampling 
#> BGGM: Finished

# note regression coefficients
samps <- posterior_samples(fit)

hist(samps[,1])

# }
```
