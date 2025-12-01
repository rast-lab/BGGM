# GGM: Estimation

Estimate the conditional (in)dependence with either an analytic solution
or efficiently sampling from the posterior distribution. These methods
were introduced in Williams (2018) . The graph is selected with
[`select.estimate`](https://rast-lab.github.io/BGGM/reference/select.estimate.md)
and then plotted with
[`plot.select`](https://rast-lab.github.io/BGGM/reference/plot.select.md).

## Usage

``` r
estimate(
  Y,
  formula = NULL,
  type = "continuous",
  mixed_type = NULL,
  analytic = FALSE,
  prior_sd = sqrt(1/3),
  iter = 5000,
  impute = FALSE,
  progress = TRUE,
  seed = NULL,
  ...
)
```

## Arguments

- Y:

  Matrix (or data frame) of dimensions *n* (observations) by *p*
  (variables).

- formula:

  An object of class [`formula`](https://rdrr.io/r/stats/formula.html).
  This allows for including control variables in the model (i.e.,
  `~ gender`). See the note for further details.

- type:

  Character string. Which type of data for `Y` ? The options include
  `continuous`, `binary`, `ordinal`, or `mixed`. Note that mixed can be
  used for data with only ordinal variables. See the note for further
  details.

- mixed_type:

  Numeric vector. An indicator of length *p* for which variables should
  be treated as ranks. (1 for rank and 0 to assume normality). The
  default is currently to treat all integer variables as ranks when
  `type = "mixed"` and `NULL` otherwise. See note for further details.

- analytic:

  Logical. Should the analytic solution be computed (default is
  `FALSE`)?

- prior_sd:

  Scale of the prior distribution, approximately the standard deviation
  of a beta distribution (defaults to sqrt(1/3)).

- iter:

  Number of iterations (posterior samples; defaults to 5000).

- impute:

  Logical. Should the missing values (`NA`) be imputed during model
  fitting (defaults to `TRUE`) ?

- progress:

  Logical. Should a progress bar be included (defaults to `TRUE`) ?

- seed:

  An integer for the random seed.

- ...:

  Currently ignored.

## Value

The returned object of class `estimate` contains a lot of information
that is used for printing and plotting the results. For users of
**BGGM**, the following are the useful objects:

- `pcor_mat` Partial correltion matrix (posterior mean).

- `post_samp` An object containing the posterior samples.

## Details

The default is to draw samples from the posterior distribution
(`analytic = FALSE`). The samples are required for computing edge
differences (see
[`ggm_compare_estimate`](https://rast-lab.github.io/BGGM/reference/ggm_compare_estimate.md)),
Bayesian R2 introduced in Gelman et al. (2019) (see
[`predictability`](https://rast-lab.github.io/BGGM/reference/predictability.md)),
etc. If the goal is to \*only\* determine the non-zero effects, this can
be accomplished by setting `analytic = TRUE`. This is particularly
useful when a fast solution is needed (see the examples in
[`ggm_compare_ppc`](https://rast-lab.github.io/BGGM/reference/ggm_compare_ppc.md))

**Controlling for Variables**:

When controlling for variables, it is assumed that `Y` includes *only*
the nodes in the GGM and the control variables. Internally, `only` the
predictors that are included in `formula` are removed from `Y`. This is
not behavior of, say, [`lm`](https://rdrr.io/r/stats/lm.html), but was
adopted to ensure users do not have to write out each variable that
should be included in the GGM. An example is provided below.

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

**Imputing Missing Values**:

Missing values are imputed with the approach described in Hoff (2009) .
The basic idea is to impute the missing values with the respective
posterior pedictive distribution, given the observed data, as the model
is being estimated. Note that the default is `TRUE`, but this ignored
when there are no missing values. If set to `FALSE`, and there are
missing values, list-wise deletion is performed with `na.omit`.

## Note

**Posterior Uncertainty**:

A key feature of **BGGM** is that there is a posterior distribution for
each partial correlation. This readily allows for visiualizing
uncertainty in the estimates. This feature works with all data types and
is accomplished by plotting the summary of the `estimate` object (i.e.,
`plot(summary(fit))`). Several examples are provided below.

**Interpretation of Conditional (In)dependence Models for Latent Data**:

See
[`BGGM-package`](https://rast-lab.github.io/BGGM/reference/BGGM-package.md)
for details about interpreting GGMs based on latent data (i.e, all data
types besides `"continuous"`)

## References

Gelman A, Goodrich B, Gabry J, Vehtari A (2019). “R-squared for Bayesian
Regression Models.” *American Statistician*, **73**(3), 307–309. ISSN
15372731.  
  
Hoff PD (2007). “Extending the rank likelihood for semiparametric copula
estimation.” *The Annals of Applied Statistics*, **1**(1), 265–283.
[doi:10.1214/07-AOAS107](https://doi.org/10.1214/07-AOAS107) .  
  
Hoff PD (2009). *A first course in Bayesian statistical methods*, volume
580. Springer.  
  
Williams DR (2018). “Bayesian Estimation for Gaussian Graphical Models:
Structure Learning, Predictability, and Network Comparisons.” *arXiv*.
[doi:10.31234/OSF.IO/X8DPR](https://doi.org/10.31234/OSF.IO/X8DPR) .

## Examples

``` r
# \donttest{
# note: iter = 250 for demonstrative purposes

#########################################
### example 1: continuous and ordinal ###
#########################################
# data
Y <- ptsd

# continuous

# fit model
fit <- estimate(Y, type = "continuous",
                iter = 250)
#> BGGM: Posterior Sampling 
#> BGGM: Finished

# summarize the partial correlations
summ <- summary(fit)

# plot the summary
plt_summ <- plot(summary(fit))

# select the graph
E <- select(fit)

# plot the selected graph
plt_E <- plot(select(fit))


# ordinal

# fit model (note + 1, due to zeros)
fit <- estimate(Y + 1, type = "ordinal",
                iter = 250)
#> BGGM: Posterior Sampling 
#> BGGM: Finished

# summarize the partial correlations
summ <- summary(fit)

# plot the summary
plt <- plot(summary(fit))

# select the graph
E <- select(fit)

# plot the selected graph
plt_E <- plot(select(fit))

##################################
## example 2: analytic solution ##
##################################
# (only continuous)

# data
Y <- ptsd

# fit model
fit <- estimate(Y, analytic = TRUE)

# summarize the partial correlations
summ <- summary(fit)

# plot summary
plt_summ <- plot(summary(fit))

# select graph
E <- select(fit)

# plot the selected graph
plt_E <- plot(select(fit))

# }
```
