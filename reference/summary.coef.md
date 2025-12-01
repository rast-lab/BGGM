# Summarize `coef` Objects

Summarize regression parameters with the posterior mean, standard
deviation, and credible interval.

## Usage

``` r
# S3 method for class 'coef'
summary(object, cred = 0.95, ...)
```

## Arguments

- object:

  An object of class `coef`.

- cred:

  Numeric. The credible interval width for summarizing the posterior
  distributions (defaults to 0.95; must be between 0 and 1).

- ...:

  Currently ignored

## Value

A list of length *p* including the summaries for each multiple
regression.

## Note

See
[`coef.estimate`](https://rast-lab.github.io/BGGM/reference/coef.estimate.md)
and
[`coef.explore`](https://rast-lab.github.io/BGGM/reference/coef.explore.md)
for examples.
