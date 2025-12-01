# Zero-Order Correlations

Estimate zero-order correlations for any type of data. Note zero-order
refers to the fact that no variables are controlled for (i.e., bivariate
correlations). To our knowledge, this is the only Bayesian
implementation in `R` that can estiamte Pearson's, tetrachoric (binary),
polychoric (ordinal with more than two cateogries), and rank based
correlation coefficients.

## Usage

``` r
zero_order_cors(
  Y,
  type = "continuous",
  iter = 5000,
  mixed_type = NULL,
  progress = TRUE
)
```

## Arguments

- Y:

  Matrix (or data frame) of dimensions *n* (observations) by *p*
  (variables).

- type:

  Character string. Which type of data for `Y` ? The options include
  `continuous`, `binary`, `ordinal`, or `mixed`. See the note for
  further details.

- iter:

  Number of iterations (posterior samples; defaults to 5000).

- mixed_type:

  Numeric vector. An indicator of length p for which varibles should be
  treated as ranks. (1 for rank and 0 to assume normality). The default
  is currently to treat all integer variables as ranks when
  `type = "mixed"` and `NULL` otherwise. See note for further details.

- progress:

  Logical. Should a progress bar be included (defaults to `TRUE`) ?

## Value

- `R` An array including the correlation matrices (of dimensions *p* by
  *p* by *iter*)

- `R_mean` Posterior mean of the correlations (of dimensions *p* by *p*)

## Details

**Mixed Type**:

The term "mixed" is somewhat of a misnomer, because the method can be
used for data including *only* continuous or *only* discrete variables.
This is based on the ranked likelihood which requires sampling the ranks
for each variable (i.e., the data is not merely transformed to ranks).
This is computationally expensive when there are many levels. For
example, with continuous data, there are as many ranks as data points!

The option `mixed_type` allows the user to determine which variable
should be treated as ranks and the "emprical" distribution is used
otherwise (Hoff 2007) . This is accomplished by specifying an indicator
vector of length *p*. A one indicates to use the ranks, whereas a zero
indicates to "ignore" that variable. By default all integer variables
are treated as ranks.

**Dealing with Errors**:

An error is most likely to arise when `type = "ordinal"`. The are two
common errors (although still rare):

- The first is due to sampling the thresholds, especially when the data
  is heavily skewed. This can result in an ill-defined matrix. If this
  occurs, we recommend to first try decreasing `prior_sd` (i.e., a more
  informative prior). If that does not work, then change the data type
  to `type = mixed` which then estimates a copula GGM (this method can
  be used for data containing **only** ordinal variable). This should
  work without a problem.

- The second is due to how the ordinal data are categorized. For
  example, if the error states that the index is out of bounds, this
  indicates that the first category is a zero. This is not allowed, as
  the first category must be one. This is addressed by adding one (e.g.,
  `Y + 1`) to the data matrix.

## Examples

``` r
# \donttest{
# note: iter = 250 for demonstrative purposes

Y <- ptsd[,1:3]

#################################
####### example 1: Pearson's ####
#################################

fit <- zero_order_cors(Y, type = "continuous",
                       iter = 250,
                       progress = FALSE)


#################################
###### example 2: polychoric ####
#################################

fit <- zero_order_cors(Y+1, type = "ordinal",
                       iter = 250,
                       progress = FALSE)


###########################
##### example 3: rank #####
###########################

fit <- zero_order_cors(Y+1, type = "mixed",
                       iter = 250,
                       progress = FALSE)

############################
## example 4: tetrachoric ##
############################

# binary data
Y <- women_math[,1:3]

fit <- zero_order_cors(Y, type = "binary",
                       iter = 250,
                       progress = FALSE)

# }
```
