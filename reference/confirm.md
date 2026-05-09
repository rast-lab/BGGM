# GGM: Confirmatory Hypothesis Testing

Confirmatory hypothesis testing in GGMs. Hypotheses are expressed as
equality and/or ineqaulity contraints on the partial correlations of
interest. Here the focus is *not* on determining the graph (see
[`explore`](https://rast-lab.github.io/BGGM/reference/explore.md)) but
testing specific hypotheses related to the conditional (in)dependence
structure. These methods were introduced in Williams and Mulder (2019) .

## Usage

``` r
confirm(
  Y,
  hypothesis,
  prior_sd = 0.5,
  formula = NULL,
  type = "continuous",
  mixed_type = NULL,
  iter = 25000,
  progress = TRUE,
  impute = TRUE,
  seed = NULL,
  ...
)
```

## Arguments

- Y:

  Matrix (or data frame) of dimensions *n* (observations) by *p*
  (variables).

- hypothesis:

  Character string. The hypothesis (or hypotheses) to be tested. See
  details.

- prior_sd:

  Numeric. Scale of the prior distribution, approximately the standard
  deviation of a beta distribution (defaults to 0.5).

- formula:

  An object of class [`formula`](https://rdrr.io/r/stats/formula.html).
  This allows for including control variables in the model (e.g.,,
  `~ gender * education`).

- type:

  Character string. Which type of data for **Y** ? The options include
  `continuous`, `binary`, `ordinal`, or `mixed`. See the note for
  further details.

- mixed_type:

  Numeric vector of length *p*. An indicator for which varibles should
  be treated as ranks. (1 for rank and 0 to assume normality). The
  default is currently (dev version) to treat all integer variables as
  ranks when `type = "mixed"` and `NULL` otherwise. See note for further
  details.

- iter:

  Number of iterations (posterior samples; defaults to 25,000).

- progress:

  Logical. Should a progress bar be included (defaults to `TRUE`) ?

- impute:

  Logicial. Should the missing values (`NA`) be imputed during model
  fitting (defaults to `TRUE`) ?

- seed:

  An integer for the random seed.

- ...:

  Currently ignored.

## Value

The returned object of class `confirm` contains a lot of information
that is used for printing and plotting the results. For users of
**BGGM**, the following are the useful objects:

- `out_hyp_prob` Posterior hypothesis probabilities.

- `info` An object of class `BF` from the R package **BFpack**.

## Details

The hypotheses can be written either with the respective column names or
numbers. For example, `1--2` denotes the relation between the variables
in column 1 and 2. Note that these must correspond to the upper
triangular elements of the correlation matrix. This is accomplished by
ensuring that the first number is smaller than the second number. This
also applies when using column names (i.e,, in reference to the column
number).

**One Hypothesis**:

To test whether some relations are larger than others, while others are
expected to be equal, this can be writting as

- `hyp <- c(1--2 > 1--3 = 1--4 > 0)`,

where there is an addition additional contraint that all effects are
expected to be positive. This is then compared to the complement.

**More Than One Hypothesis**:

The above hypothesis can also be compared to, say, a null model by using
";" to seperate the hypotheses, for example,

- `hyp <- c(1--2 > 1--3 = 1--4 > 0; 1--2 = 1--3 = 1--4 = 0)`.

Any number of hypotheses can be compared this way.

**Using "&"**

It is also possible to include `&`. This allows for testing one
constraint **and** another contraint as one hypothesis.

- `hyp <- c("A1--A2 > A1--A2 & A1--A3 = A1--A3")`

Of course, it is then possible to include additional hypotheses by
separating them with ";". Note also that the column names were used in
this example (e.g., `A1--A2` is the relation between those nodes).

**Testing Sums**

It might also be interesting to test the sum of partial correlations.
For example, that the sum of specific relations is larger than the sum
of other relations. This can be written as

- `hyp <- c("A1--A2 + A1--A3 > A1--A4 + A1--A5; A1--A2 + A1--A3 = A1--A4 + A1--A5")`

**Potential Delays**:

There is a chance for a potentially long delay from the time the
progress bar finishes to when the function is done running. This occurs
when the hypotheses require further sampling to be tested, for example,
when grouping relations `c("(A1--A2, A1--A3) > (A1--A4, A1--A5)"`. This
is not an error.

**Controlling for Variables**:

When controlling for variables, it is assumed that `Y` includes *only*
the nodes in the GGM and the control variables. Internally, `only` the
predictors that are included in `formula` are removed from `Y`. This is
not behavior of, say, [`lm`](https://rdrr.io/r/stats/lm.html), but was
adopted to ensure users do not have to write out each variable that
should be included in the GGM. An example is provided below.

**Mixed Type**:

The term "mixed" is somewhat of a misnomer, because the method can be
used for data including *only* continuous or *only* discrete variables
(Hoff 2007) . This is based on the ranked likelihood which requires
sampling the ranks for each variable (i.e., the data is not merely
transformed to ranks). This is computationally expensive when there are
many levels. For example, with continuous data, there are as many ranks
as data points!

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

## Note

**"Default" Prior**:

In Bayesian statistics, a default Bayes factor needs to have several
properties. I refer interested users to section 2.2 in Dablander et al.
(2020) . In Williams and Mulder (2019) , some of these propteries were
investigated (e.g., model selection consistency). That said, we would
not consider this a "default" or "automatic" Bayes factor and thus we
encourage users to perform sensitivity analyses by varying the scale of
the prior distribution.

Furthermore, it is important to note there is no "correct" prior and,
also, there is no need to entertain the possibility of a "true" model.
Rather, the Bayes factor can be interpreted as which hypothesis best
(relative to each other) predicts the observed data (Section 3.2 in Kass
and Raftery 1995) .

**Interpretation of Conditional (In)dependence Models for Latent Data**:

See
[`BGGM-package`](https://rast-lab.github.io/BGGM/reference/BGGM-package.md)
for details about interpreting GGMs based on latent data (i.e, all data
types besides `"continuous"`)

## References

Dablander F, Bergh Dvd, Ly A, Wagenmakers E (2020). “Default Bayes
Factors for Testing the (In) equality of Several Population Variances.”
*arXiv preprint arXiv:2003.06278*.  
  
Hoff PD (2007). “Extending the rank likelihood for semiparametric copula
estimation.” *The Annals of Applied Statistics*, **1**(1), 265–283.
[doi:10.1214/07-AOAS107](https://doi.org/10.1214/07-AOAS107) .  
  
Kass RE, Raftery AE (1995). “Bayes Factors.” *Journal of the American
Statistical Association*, **90**(430), 773–795.  
  
Williams DR, Mulder J (2019). “Bayesian Hypothesis Testing for Gaussian
Graphical Models: Conditional Independence and Order Constraints.”
*PsyArXiv*.
[doi:10.31234/osf.io/ypxd8](https://doi.org/10.31234/osf.io/ypxd8) .

## Examples

``` r
# \donttest{
# note: iter = 250 for demonstrative purposes

##########################
### example 1: cheating ##
##########################
# Here a true hypothesis is tested,
# which shows the method works nicely
# (peeked at partials beforehand)

# data
Y <- BGGM::bfi[,1:10]

hypothesis <- c("A1--A2 < A1--A3 < A1--A4 = A1--A5")

# test cheat
test_cheat <-  confirm(Y = Y,
                       type = "continuous",
                       hypothesis  = hypothesis,
                       iter = 250,
                       progress = FALSE)

# print (probabilty of nearly 1 !)
test_cheat
#> BGGM: Bayesian Gaussian Graphical Models 
#> Type: continuous 
#> --- 
#> Posterior Samples: 250 
#> Observations (n): 2800 
#> Variables (p): 10 
#> Delta: 3 
#> --- 
#> Call:
#> confirm(Y = Y, hypothesis = hypothesis, type = "continuous", 
#>     iter = 250, progress = FALSE)
#> --- 
#> Hypotheses: 
#> 
#> H1: A1--A2<A1--A3<A1--A4=A1--A5
#> H2: complement
#> --- 
#> Posterior prob: 
#> 
#> p(H1|data) = 0.996
#> p(H2|data) = 0.004
#> --- 
#> Bayes factor matrix: 
#>       H1      H2
#> H1 1.000 229.001
#> H2 0.004   1.000
#> --- 
#> note: equal hypothesis prior probabilities
# }
```
