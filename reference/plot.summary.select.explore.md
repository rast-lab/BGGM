# Plot `summary.select.explore` Objects

Visualize the posterior hypothesis probabilities.

## Usage

``` r
# S3 method for class 'summary.select.explore'
plot(x, size = 2, color = "black", ...)
```

## Arguments

- x:

  An object of class `summary.select.explore`

- size:

  Numeric. The size for the points (defaults to 2).

- color:

  Character string. The Color for the points

- ...:

  Currently ignored

## Value

A `ggplot` object

## Examples

``` r
# \donttest{
#  data
Y <- bfi[,1:10]

# fit model
fit <- explore(Y, iter = 250,
               progress = FALSE)

# edge set
E <- select(fit,
            alternative = "exhaustive")

plot(summary(E))


# }
```
