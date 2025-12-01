# Plot `summary.estimate` Objects

Visualize the posterior distributions for each partial correlation.

## Usage

``` r
# S3 method for class 'summary.estimate'
plot(x, color = "black", size = 2, width = 0, ...)
```

## Arguments

- x:

  An object of class `summary.estimate`

- color:

  Character string. The color for the error bars. (defaults to
  `"black"`).

- size:

  Numeric. The size for the points (defaults to `2`).

- width:

  Numeric. The width of error bar ends (defaults to `0`).

- ...:

  Currently ignored

## Value

A `ggplot` object.

## See also

[`estimate`](https://rast-lab.github.io/BGGM/reference/estimate.md)

## Examples

``` r
# \donttest{
# data
Y <- ptsd[,1:5]

fit <- estimate(Y, iter = 250,
                progress = FALSE)


plot(summary(fit))


# }
```
