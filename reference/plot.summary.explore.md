# Plot `summary.explore` Objects

Visualize the posterior distributions for each partial correlation.

## Usage

``` r
# S3 method for class 'summary.explore'
plot(x, color = "black", size = 2, width = 0, ...)
```

## Arguments

- x:

  An object of class `summary.explore`

- color:

  Character string. The color for the error bars. (defaults to
  `"black"`).

- size:

  Numeric. The size for the points (defaults to `2`).

- width:

  Numeric. The width of error bar ends (defaults to `0` ).

- ...:

  Currently ignored

## Value

A `ggplot` object

## See also

[`explore`](https://rast-lab.github.io/BGGM/reference/explore.md)

## Examples

``` r
# \donttest{
# note: iter = 250 for demonstrative purposes

Y <- ptsd[,1:5]

fit <- explore(Y, iter = 250,
               progress = FALSE)

plt <- plot(summary(fit))

plt

# }
```
