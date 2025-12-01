# Plot `summary.ggm_compare_explore` Objects

Visualize the posterior hypothesis probabilities.

## Usage

``` r
# S3 method for class 'summary.ggm_compare_explore'
plot(x, size = 2, color = "black", ...)
```

## Arguments

- x:

  An object of class `summary.ggm_compare_explore`

- size:

  Numeric. The size of the points (defaults to 2).

- color:

  Character string. The color of the points (defaults to `"black"`).

- ...:

  Currently ignored.

## Value

A `ggplot` object

## See also

[`ggm_compare_explore`](https://rast-lab.github.io/BGGM/reference/ggm_compare_explore.md)

## Examples

``` r
# \donttest{
# note: iter = 250 for demonstrative purposes

# data
Y <- bfi[complete.cases(bfi),]

# males and females
Ymale <- subset(Y, gender == 1,
                   select = -c(gender,
                               education))[,1:10]

Yfemale <- subset(Y, gender == 2,
                     select = -c(gender,
                                 education))[,1:10]

##########################
### example 1: ordinal ###
##########################

# fit model
fit <- ggm_compare_explore(Ymale,  Yfemale,
                           type = "ordinal",
                           iter = 250,
                           progress = FALSE)
# summary
summ <- summary(fit)

plot(summ)

# }
```
