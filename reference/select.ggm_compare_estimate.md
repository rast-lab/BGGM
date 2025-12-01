# Graph Selection for `ggm_compare_estimate` Objects

Provides the selected graph (of differences) based on credible intervals
for the partial correlations that did not contain zero (Williams 2018) .

## Usage

``` r
# S3 method for class 'ggm_compare_estimate'
select(object, cred = 0.95, ...)
```

## Arguments

- object:

  An object of class `estimate.default`.

- cred:

  Numeric. The credible interval width for selecting the graph (defaults
  to 0.95; must be between 0 and 1).

- ...:

  not currently used

## Value

The returned object of class `select.ggm_compare_estimate` contains a
lot of information that is used for printing and plotting the results.
For users of **BGGM**, the following are the useful objects:

- `mean_diff` A list of matrices for each group comparsion (partial
  correlation differences).

- `pcor_adj` A list of weighted adjacency matrices for each group
  comparsion.

- `adj` A list of adjacency matrices for each group comparsion.

## Examples

``` r
# \donttest{
# note: iter = 250 for demonstrative purposes
##################
### example 1: ###
##################
# data
Y <- bfi

# males and females
Ymale <- subset(Y, gender == 1,
               select = -c(gender,
                           education))

Yfemale <- subset(Y, gender == 2,
                  select = -c(gender,
                              education))

# fit model
fit <- ggm_compare_estimate(Ymale, Yfemale,
                           type = "continuous",
                           iter = 250,
                           progress = FALSE)


E <- select(fit)

# }
```
