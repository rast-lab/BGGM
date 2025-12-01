# Compute Correlations from the Partial Correlations

Convert the partial correlation matrices into correlation matrices. To
our knowledge, this is the only Bayesian implementation in `R` that can
estiamte Pearson's, tetrachoric (binary), polychoric (ordinal with more
than two cateogries), and rank based correlation coefficients.

## Usage

``` r
pcor_to_cor(object, iter = NULL)
```

## Arguments

- object:

  An object of class `estimate` or `explore`

- iter:

  numeric. How many iterations (i.e., posterior samples) should be used
  ? The default uses all of the samples, but note that this can take a
  long time with large matrices.

## Value

- `R` An array including the correlation matrices (of dimensions *p* by
  *p* by *iter*)

- `R_mean` Posterior mean of the correlations (of dimensions *p* by *p*)

## Note

The 'default' prior distributions are specified for partial correlations
in particular. This means that the implied prior distribution will not
be the same for the correlations.

## Examples

``` r
# \donttest{
# note: iter = 250 for demonstrative purposes

# data
Y <- BGGM::ptsd

#########################
###### continuous #######
#########################

# estimate the model
fit <- estimate(Y, iter = 250,
                progress = FALSE)

# compute correlations
cors <- pcor_to_cor(fit)


#########################
###### ordinal  #########
#########################

# first level must be 1 !
Y <- Y + 1

# estimate the model
fit <- estimate(Y, type =  "ordinal",
                iter = 250,
                progress = FALSE)

# compute correlations
cors <- pcor_to_cor(fit)


#########################
#######   mixed    ######
#########################

# rank based correlations

# estimate the model
fit <- estimate(Y, type =  "mixed",
                iter = 250,
                progress = FALSE)

# compute correlations
cors <- pcor_to_cor(fit)
# }
```
