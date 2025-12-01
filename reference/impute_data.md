# Obtain Imputed Datasets

Impute missing values, assuming a multivariate normal distribution, with
the posterior predictive distribution. For binary, ordinal, and mixed (a
combination of discrete and continuous) data, the values are first
imputed for the latent data and then converted to the original scale.

## Usage

``` r
impute_data(
  Y,
  type = "continuous",
  lambda = NULL,
  mixed_type = NULL,
  iter = 1000,
  progress = TRUE
)
```

## Arguments

- Y:

  Matrix (or data frame) of dimensions *n* (observations) by *p*
  (variables).

- type:

  Character string. Which type of data for `Y` ? The options include
  `continuous`, `binary`, `ordinal`, or `mixed`. Note that mixed can be
  used for data with only ordinal variables. See the note for further
  details.

- lambda:

  Numeric. A regularization parameter, which defaults to p + 2. A larger
  value results in more shrinkage.

- mixed_type:

  Numeric vector. An indicator of length *p* for which variables should
  be treated as ranks. (1 for rank and 0 to assume the observed marginal
  distribution). The default is currently to treat all integer variables
  as ranks when `type = "mixed"` and `NULL` otherwise. See note for
  further details.

- iter:

  Number of iterations (posterior samples; defaults to 1000).

- progress:

  Logical. Should a progress bar be included (defaults to `TRUE`) ?

## Value

An object of class `mvn_imputation`:

- `imputed_datasets` An array including the imputed datasets.

## Details

Missing values are imputed with the approach described in Hoff (2009) .
The basic idea is to impute the missing values with the respective
posterior pedictive distribution, given the observed data, as the model
is being estimated. Note that the default is `TRUE`, but this ignored
when there are no missing values. If set to `FALSE`, and there are
missing values, list-wise deletion is performed with `na.omit`.

## References

Hoff PD (2009). *A first course in Bayesian statistical methods*, volume
580. Springer.

## Examples

``` r
# \donttest{
# obs
n <- 5000

# n missing
n_missing <- 1000

# variables
p <- 16

# data
Y <- MASS::mvrnorm(n, rep(0, p), ptsd_cor1)

# for checking
Ymain <- Y

# all possible indices
indices <- which(matrix(0, n, p) == 0,
                 arr.ind = TRUE)

# random sample of 1000 missing values
na_indices <- indices[sample(5:nrow(indices),
                             size = n_missing,
                             replace = FALSE),]

# fill with NA
Y[na_indices] <- NA

# missing = 1
Y_miss <- ifelse(is.na(Y), 1, 0)

# true values (to check)
true <- unlist(sapply(1:p, function(x)
        Ymain[which(Y_miss[,x] == 1),x] ))

# impute
fit_missing <- impute_data(Y, progress = FALSE, iter = 250)

# impute
fit_missing <- impute_data(Y,
                           progress = TRUE,
                           iter = 250)
#> BGGM: Imputing
#> BGGM: Finished

# }
```
