# Plot `summary.ggm_compare_estimate` Objects

Visualize the posterior distribution differences.

## Usage

``` r
# S3 method for class 'summary.ggm_compare_estimate'
plot(x, color = "black", size = 2, width = 0, ...)
```

## Arguments

- x:

  An object of class `ggm_compare_estimate`.

- color:

  Character string. The color of the points (defaults to `"black"`).

- size:

  Numeric. The size of the points (defaults to 2).

- width:

  Numeric. The width of error bar ends (defaults to `0`).

- ...:

  Currently ignored.

## Value

An object of class `ggplot`

## See also

[`ggm_compare_estimate`](https://rast-lab.github.io/BGGM/reference/ggm_compare_estimate.md)

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

# fit model
fit <- ggm_compare_estimate(Ymale,  Yfemale,
                            type = "ordinal",
                            iter = 250,
                            prior_sd = 0.25,
                            progress = FALSE)
#> Warning: imputation during model fitting is
#> currently only implemented for 'continuous' data.
#> Warning: imputation during model fitting is
#> currently only implemented for 'continuous' and 'mixed' data.
#> Warning: imputation during model fitting is
#> currently only implemented for 'continuous' and 'mixed' data.

plot(summary(fit))
#> [[1]]

#> 
# }
```
