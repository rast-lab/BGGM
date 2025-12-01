# Graph selection for `ggm_compare_explore` Objects

Provides the selected graph (of differences) based on the Bayes factor
(Williams et al. 2020) .

## Usage

``` r
# S3 method for class 'ggm_compare_explore'
select(object, BF_cut = 3, ...)
```

## Arguments

- object:

  An object of class `ggm_compare_explore`.

- BF_cut:

  Numeric. Threshold for including an edge (defaults to 3).

- ...:

  Currently ignored.

## Value

The returned object of class `select.ggm_compare_explore` contains a lot
of information that is used for printing and plotting the results. For
users of **BGGM**, the following are the useful objects:

- `adj_10` Adjacency matrix for which there was evidence for a
  difference.

- `adj_10` Adjacency matrix for which there was evidence for a null
  relation

- `pcor_mat_10` Selected partial correlation matrix (weighted adjacency;
  only for two groups).

## See also

[`explore`](https://rast-lab.github.io/BGGM/reference/explore.md) and
[`ggm_compare_explore`](https://rast-lab.github.io/BGGM/reference/ggm_compare_explore.md)
for several examples.

## Examples

``` r
# \donttest{

##################
### example 1: ###
##################
# data
Y <- bfi

# males and females
Ymale <- subset(Y, gender == 1,
                   select = -c(gender,
                               education))[,1:10]

Yfemale <- subset(Y, gender == 2,
                     select = -c(gender,
                                 education))[,1:10]

# fit model
fit <- ggm_compare_explore(Ymale, Yfemale,
                           iter = 250,
                           type = "continuous",
                           progress = FALSE)


E <- select(fit, post_prob = 0.50)

# }
```
