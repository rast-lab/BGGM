# GGM Compare: Estimate

Compare partial correlations that are estimated from any number of
groups. This method works for continuous, binary, ordinal, and mixed
data (a combination of categorical and continuous variables). The
approach (i.e., a difference between posterior distributions) was
described in Williams (2018) .

## Usage

``` r
ggm_compare_estimate(
  ...,
  formula = NULL,
  type = "continuous",
  mixed_type = NULL,
  analytic = FALSE,
  prior_sd = sqrt(1/3),
  iter = 5000,
  impute = TRUE,
  progress = TRUE,
  seed = 1
)
```

## Arguments

- ...:

  Matrices (or data frames) of dimensions *n* (observations) by *p*
  (variables). Requires at least two.

- formula:

  An object of class [`formula`](https://rdrr.io/r/stats/formula.html).
  This allows for including control variables in the model (i.e.,
  `~ gender`). See the note for further details.

- type:

  Character string. Which type of data for **Y** ? The options include
  `continuous`, `binary`, `ordinal`, or `continuous`. See the note for
  further details.

- mixed_type:

  Numeric vector. An indicator of length *p* for which varibles should
  be treated as ranks. (1 for rank and 0 to use the 'empirical' or
  observed distribution). The default is currently to treat all integer
  variables as ranks when `type = "mixed"` and `NULL` otherwise. See
  note for further details.

- analytic:

  Logical. Should the analytic solution be computed (default is
  `FALSE`)? This is only available for continous data. Note that if
  `type = "mixed"` and `analytic = TRUE`, the data will automatically be
  treated as continuous.

- prior_sd:

  The scale of the prior distribution (centered at zero), in reference
  to a beta distribtuion (defaults to sqrt(1/3)). See note for further
  details.

- iter:

  Number of iterations (posterior samples; defaults to 5000).

- impute:

  Logicial. Should the missing values (`NA`) be imputed during model
  fitting (defaults to `TRUE`) ?

- progress:

  Logical. Should a progress bar be included (defaults to `TRUE`) ?

- seed:

  An integer for the random seed.

## Value

A list of class `ggm_compare_estimate` containing:

- `pcor_diffs` partial correlation differences (posterior distribution)

- `p` number of variable

- `info` list containing information about each group (e.g., sample
  size, etc.)

- `iter` number of posterior samples

- `call` `match.call`

## Details

This function can be used to compare the partial correlations for any
number of groups. This is accomplished with pairwise comparisons for
each relation. In the case of three groups, for example, group 1 and
group 2 are compared, then group 1 and group 3 are compared, and then
group 2 and group 3 are compared. There is a full distibution for each
difference that can be summarized (i.e.,
[`summary.ggm_compare_estimate`](https://rast-lab.github.io/BGGM/reference/summary.ggm_compare_estimate.md))
and then visualized (i.e.,
[`plot.summary.ggm_compare_estimate`](https://rast-lab.github.io/BGGM/reference/plot.summary.ggm_compare_estimate.md)).
The graph of difference is selected with
[`select.ggm_compare_estimate`](https://rast-lab.github.io/BGGM/reference/select.ggm_compare_estimate.md)).

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
otherwise. This is accomplished by specifying an indicator vector of
length *p*. A one indicates to use the ranks, whereas a zero indicates
to "ignore" that variable. By default all integer variables are handled
as ranks.

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

**Mixed Data**:

The mixed data approach was introduced in Hoff (2007) (our paper
describing an extension to Bayesian hypothesis testing if forthcoming).
This is a semi-paramateric copula model based on the ranked likelihood.
This is computationally expensive when treating continuous data as
ranks. The current default is to treat only integer data as ranks. This
should of course be adjusted for continous data that is skewed. This can
be accomplished with the argument `mixed_type`. A `1` in the numeric
vector of length *p*indicates to treat that respective node as a rank
(corresponding to the column number) and a zero indicates to use the
observed (or "emprical") data.

It is also important to note that `type = "mixed"` is not restricted to
mixed data (containing a combination of categorical and continuous): all
the nodes can be ordinal or continuous (but again this will take some
time).

**Interpretation of Conditional (In)dependence Models for Latent Data**:

See
[`BGGM-package`](https://rast-lab.github.io/BGGM/reference/BGGM-package.md)
for details about interpreting GGMs based on latent data (i.e, all data
types besides `"continuous"`)

**Additional GGM Compare Methods**

Bayesian hypothesis testing is implemented in
[`ggm_compare_explore`](https://rast-lab.github.io/BGGM/reference/ggm_compare_explore.md)
and
[`ggm_compare_confirm`](https://rast-lab.github.io/BGGM/reference/ggm_compare_confirm.md)
(Williams and Mulder 2019) . The latter allows for confirmatory
hypothesis testing. An approach based on a posterior predictive check is
implemented in
[`ggm_compare_ppc`](https://rast-lab.github.io/BGGM/reference/ggm_compare_ppc.md)
(Williams et al. 2020) . This provides a 'global' test for comparing the
entire GGM and a 'nodewise' test for comparing each variable in the
network Williams (2018) .

## References

Hoff PD (2007). “Extending the rank likelihood for semiparametric copula
estimation.” *The Annals of Applied Statistics*, **1**(1), 265–283.
[doi:10.1214/07-AOAS107](https://doi.org/10.1214/07-AOAS107) .  
  
Hoff PD (2009). *A first course in Bayesian statistical methods*, volume
580. Springer.  
  
Williams DR (2018). “Bayesian Estimation for Gaussian Graphical Models:
Structure Learning, Predictability, and Network Comparisons.” *arXiv*.
[doi:10.31234/OSF.IO/X8DPR](https://doi.org/10.31234/OSF.IO/X8DPR) .  
  
Williams DR, Mulder J (2019). “Bayesian Hypothesis Testing for Gaussian
Graphical Models: Conditional Independence and Order Constraints.”
*PsyArXiv*.
[doi:10.31234/osf.io/ypxd8](https://doi.org/10.31234/osf.io/ypxd8) .  
  
Williams DR, Rast P, Pericchi LR, Mulder J (2020). “Comparing Gaussian
graphical models with the posterior predictive distribution and Bayesian
model selection.” *Psychological Methods*.
[doi:10.1037/met0000254](https://doi.org/10.1037/met0000254) .

## Examples

``` r
# \donttest{
# note: iter = 250 for demonstrative purposes

# data: Remove missings for "ordinal"
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
                           progress = FALSE)
#> Warning: imputation during model fitting is
#> currently only implemented for 'continuous' data.
#> Warning: imputation during model fitting is
#> currently only implemented for 'continuous' and 'mixed' data.
#> Warning: imputation during model fitting is
#> currently only implemented for 'continuous' and 'mixed' data.

###########################
### example 2: analytic ###
###########################
# only continuous

# fit model
fit <- ggm_compare_estimate(Ymale, Yfemale,
                            analytic = TRUE)

# summary
summ <- summary(fit)

# plot summary
plt_summ <- plot(summary(fit))

# select
E <- select(fit)

# plot select
plt_E <- plot(select(fit))

# }
```
