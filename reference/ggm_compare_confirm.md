# GGM Compare: Confirmatory Hypothesis Testing

Confirmatory hypothesis testing for comparing GGMs. Hypotheses are
expressed as equality and/or ineqaulity contraints on the partial
correlations of interest. Here the focus is *not* on determining the
graph (see
[`explore`](https://rast-lab.github.io/BGGM/reference/explore.md)) but
testing specific hypotheses related to the conditional (in)dependence
structure. These methods were introduced in Williams and Mulder (2019)
and in Williams et al. (2020)

## Usage

``` r
ggm_compare_confirm(
  ...,
  hypothesis,
  formula = NULL,
  type = "continuous",
  mixed_type = NULL,
  prior_sd = 0.5,
  iter = 25000,
  impute = TRUE,
  progress = TRUE,
  seed = NULL
)
```

## Arguments

- ...:

  At least two matrices (or data frame) of dimensions *n* (observations)
  by *p* (nodes).

- hypothesis:

  Character string. The hypothesis (or hypotheses) to be tested. See
  notes for futher details.

- formula:

  an object of class [`formula`](https://rdrr.io/r/stats/formula.html).
  This allows for including control variables in the model (i.e.,
  `~ gender`).

- type:

  Character string. Which type of data for `Y` ? The options include
  `continuous`, `binary`, `ordinal`, or `mixed`. Note that mixed can be
  used for data with only ordinal variables. See the note for further
  details.

- mixed_type:

  numeric vector. An indicator of length p for which varibles should be
  treated as ranks. (1 for rank and 0 to assume normality). The default
  is currently (dev version) to treat all integer variables as ranks
  when `type = "mixed"` and `NULL` otherwise. See note for further
  details.

- prior_sd:

  Numeric. The scale of the prior distribution (centered at zero), in
  reference to a beta distribtuion (defaults to 0.5).

- iter:

  Number of iterations (posterior samples; defaults to 25,000).

- impute:

  Logicial. Should the missing values (`NA`) be imputed during model
  fitting (defaults to `TRUE`) ?

- progress:

  Logical. Should a progress bar be included (defaults to `TRUE`) ?

- seed:

  An integer for the random seed.

## Value

The returned object of class `confirm` contains a lot of information
that is used for printing and plotting the results. For users of
**BGGM**, the following are the useful objects:

- `out_hyp_prob` Posterior hypothesis probabilities.

- `info` An object of class `BF` from the R package **BFpack** (Mulder
  et al. 2019)

## Details

The hypotheses can be written either with the respective column names or
numbers. For example, `g1_1--2` denotes the relation between the
variables in column 1 and 2 for group 1. The `g1_` is required and the
only difference from
[`confirm`](https://rast-lab.github.io/BGGM/reference/confirm.md) (one
group). Note that these must correspond to the upper triangular elements
of the correlation matrix. This is accomplished by ensuring that the
first number is smaller than the second number. This also applies when
using column names (i.e,, in reference to the column number).

**One Hypothesis**:

To test whether a relation in larger in one group, while both are
expected to be positive, this can be written as

- `hyp <- c(g1_1--2 > g2_1--2 > 0)`

This is then compared to the complement.

**More Than One Hypothesis**:

The above hypothesis can also be compared to, say, a null model by using
";" to seperate the hypotheses, for example,

- `hyp <- c(g1_1--2 > g2_1--2 > 0; g1_1--2 = g2_1--2 = 0)`.

Any number of hypotheses can be compared this way.

**Using "&"**

It is also possible to include `&`. This allows for testing one
constraint **and** another contraint as one hypothesis.

- `hyp <- c("g1_A1--A2 > g2_A1--A2 & g1_A1--A3 = g2_A1--A3")`

Of course, it is then possible to include additional hypotheses by
separating them with ";".

**Testing Sums**

It might also be interesting to test the sum of partial correlations.
For example, that the sum of specific relations in one group is larger
than the sum in another group.

- `hyp <- c("g1_A1--A2 + g1_A1--A3 > g2_A1--A2 + g2_A1--A3; g1_A1--A2 + g1_A1--A3 = g2_A1--A2 + g2_A1--A3")`

**Potential Delays**:

There is a chance for a potentially long delay from the time the
progress bar finishes to when the function is done running. This occurs
when the hypotheses require further sampling to be tested, for example,
when grouping relations
`c("(g1_A1--A2, g2_A2--A3) > (g2_A1--A2, g2_A2--A3)"`. This is not an
error.

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

**Imputing Missing Values**:

Missing values are imputed with the approach described in Hoff (2009) .
The basic idea is to impute the missing values with the respective
posterior pedictive distribution, given the observed data, as the model
is being estimated. Note that the default is `TRUE`, but this ignored
when there are no missing values. If set to `FALSE`, and there are
missing values, list-wise deletion is performed with `na.omit`.

## Note

**"Default" Prior**:

In Bayesian statistics, a default Bayes factor needs to have several
properties. I refer interested users to section 2.2 in Dablander et al.
(2020) . In Williams and Mulder (2019) , some of these propteries were
investigated (e.g., model selection consistency). That said, we would
not consider this a "default" or "automatic" Bayes factor and thus we
encourage users to perform sensitivity analyses by varying the scale of
the prior distribution (`prior_sd`).

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
  
Hoff PD (2009). *A first course in Bayesian statistical methods*, volume
580. Springer.  
  
Kass RE, Raftery AE (1995). “Bayes Factors.” *Journal of the American
Statistical Association*, **90**(430), 773–795.  
  
Mulder J, Gu X, Olsson-Collentine A, Tomarken A, Böing-Messing F,
Hoijtink H, Meijerink M, Williams DR, Menke J, Fox J, others (2019).
“BFpack: Flexible Bayes Factor Testing of Scientific Theories in R.”
*arXiv preprint arXiv:1911.07728*.  
  
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

# data
Y <- bfi

###############################
#### example 1: continuous ####
###############################

# males
Ymale   <- subset(Y, gender == 1,
                  select = -c(education,
                              gender))[,1:5]


# females
Yfemale <- subset(Y, gender == 2,
                     select = -c(education,
                                 gender))[,1:5]

 # exhaustive
 hypothesis <- c("g1_A1--A2 >  g2_A1--A2;
                  g1_A1--A2 <  g2_A1--A2;
                  g1_A1--A2 =  g2_A1--A2")

# test hyp
test <- ggm_compare_confirm(Ymale,  Yfemale,
                            hypothesis = hypothesis,
                            iter = 250,
                            progress = FALSE)

# print (evidence not strong)
test
#> BGGM: Bayesian Gaussian Graphical Models 
#> Type: continuous 
#> --- 
#> Posterior Samples: 250 
#>   Group 1: 896 
#>   Group 2: 1813 
#> Variables (p): 5 
#> Relations: 10 
#> Delta: 3 
#> --- 
#> Call:
#> ggm_compare_confirm(Ymale, Yfemale, hypothesis = hypothesis, 
#>     iter = 250, progress = FALSE)
#> --- 
#> Hypotheses: 
#> 
#> H1: g1_A1--A2>g2_A1--A2
#> H2: g1_A1--A2<g2_A1--A2
#> H3: g1_A1--A2=g2_A1--A2
#> --- 
#> Posterior prob: 
#> 
#> p(H1|data) = 0.119
#> p(H2|data) = 0.017
#> p(H3|data) = 0.863
#> --- 
#> Bayes factor matrix: 
#>       H1     H2    H3
#> H1 1.000  6.808 0.138
#> H2 0.147  1.000 0.020
#> H3 7.254 49.379 1.000
#> --- 
#> note: equal hypothesis prior probabilities

#########################################
#### example 2: sensitivity to prior ####
#########################################
# continued from example 1

# decrease prior SD
test <- ggm_compare_confirm(Ymale,
                            Yfemale,
                            prior_sd = 0.1,
                            hypothesis = hypothesis,
                            iter = 250,
                            progress = FALSE)

# print
test
#> BGGM: Bayesian Gaussian Graphical Models 
#> Type: continuous 
#> --- 
#> Posterior Samples: 250 
#>   Group 1: 896 
#>   Group 2: 1813 
#> Variables (p): 5 
#> Relations: 10 
#> Delta: 99 
#> --- 
#> Call:
#> ggm_compare_confirm(Ymale, Yfemale, hypothesis = hypothesis, 
#>     prior_sd = 0.1, iter = 250, progress = FALSE)
#> --- 
#> Hypotheses: 
#> 
#> H1: g1_A1--A2>g2_A1--A2
#> H2: g1_A1--A2<g2_A1--A2
#> H3: g1_A1--A2=g2_A1--A2
#> --- 
#> Posterior prob: 
#> 
#> p(H1|data) = 0.388
#> p(H2|data) = 0.055
#> p(H3|data) = 0.557
#> --- 
#> Bayes factor matrix: 
#>       H1     H2    H3
#> H1 1.000  7.064 0.696
#> H2 0.142  1.000 0.099
#> H3 1.436 10.143 1.000
#> --- 
#> note: equal hypothesis prior probabilities

# indecrease prior SD
test <- ggm_compare_confirm(Ymale,
                            Yfemale,
                            prior_sd = 0.28,
                            hypothesis = hypothesis,
                            iter = 250,
                            progress = FALSE)

# print
test
#> BGGM: Bayesian Gaussian Graphical Models 
#> Type: continuous 
#> --- 
#> Posterior Samples: 250 
#>   Group 1: 896 
#>   Group 2: 1813 
#> Variables (p): 5 
#> Relations: 10 
#> Delta: 11.7551 
#> --- 
#> Call:
#> ggm_compare_confirm(Ymale, Yfemale, hypothesis = hypothesis, 
#>     prior_sd = 0.28, iter = 250, progress = FALSE)
#> --- 
#> Hypotheses: 
#> 
#> H1: g1_A1--A2>g2_A1--A2
#> H2: g1_A1--A2<g2_A1--A2
#> H3: g1_A1--A2=g2_A1--A2
#> --- 
#> Posterior prob: 
#> 
#> p(H1|data) = 0.272
#> p(H2|data) = 0.026
#> p(H3|data) = 0.703
#> --- 
#> Bayes factor matrix: 
#>       H1     H2    H3
#> H1 1.000 10.563 0.387
#> H2 0.095  1.000 0.037
#> H3 2.585 27.308 1.000
#> --- 
#> note: equal hypothesis prior probabilities

################################
#### example 3: mixed data #####
################################

hypothesis <- c("g1_A1--A2 >  g2_A1--A2;
                 g1_A1--A2 <  g2_A1--A2;
                 g1_A1--A2 =  g2_A1--A2")

# test (1000 for example)
test <- ggm_compare_confirm(Ymale,
                            Yfemale,
                            type = "mixed",
                            hypothesis = hypothesis,
                            iter = 250,
                            progress = FALSE)
#> Warning: imputation during model fitting is
#> currently only implemented for 'continuous' data.

# print
test
#> BGGM: Bayesian Gaussian Graphical Models 
#> Type: mixed 
#> --- 
#> Posterior Samples: 250 
#>   Group 1: 896 
#>   Group 2: 1813 
#> Variables (p): 5 
#> Relations: 10 
#> Delta: 3 
#> --- 
#> Call:
#> ggm_compare_confirm(Ymale, Yfemale, hypothesis = hypothesis, 
#>     type = "mixed", iter = 250, progress = FALSE)
#> --- 
#> Hypotheses: 
#> 
#> H1: g1_A1--A2>g2_A1--A2
#> H2: g1_A1--A2<g2_A1--A2
#> H3: g1_A1--A2=g2_A1--A2
#> --- 
#> Posterior prob: 
#> 
#> p(H1|data) = 0.048
#> p(H2|data) = 0.047
#> p(H3|data) = 0.905
#> --- 
#> Bayes factor matrix: 
#>        H1     H2    H3
#> H1  1.000  1.006 0.052
#> H2  0.994  1.000 0.052
#> H3 19.049 19.161 1.000
#> --- 
#> note: equal hypothesis prior probabilities

##############################
##### example 4: control #####
##############################
# control for education

# data
Y <- bfi

# males
Ymale   <- subset(Y, gender == 1,
                  select = -c(gender))[,c(1:5, 26)]

# females
Yfemale <- subset(Y, gender == 2,
                  select = -c(gender))[,c(1:5, 26)]

# test
test <- ggm_compare_confirm(Ymale,
                             Yfemale,
                             formula = ~ education,
                             hypothesis = hypothesis,
                             iter = 250,
                             progress = FALSE)
# print
test
#> BGGM: Bayesian Gaussian Graphical Models 
#> Type: continuous 
#> --- 
#> Posterior Samples: 250 
#>   Group 1: 817 
#>   Group 2: 1676 
#> Variables (p): 5 
#> Relations: 10 
#> Delta: 3 
#> --- 
#> Call:
#> ggm_compare_confirm(Ymale, Yfemale, hypothesis = hypothesis, 
#>     formula = ~education, iter = 250, progress = FALSE)
#> --- 
#> Hypotheses: 
#> 
#> H1: g1_A1--A2>g2_A1--A2
#> H2: g1_A1--A2<g2_A1--A2
#> H3: g1_A1--A2=g2_A1--A2
#> --- 
#> Posterior prob: 
#> 
#> p(H1|data) = 0.022
#> p(H2|data) = 0.073
#> p(H3|data) = 0.905
#> --- 
#> Bayes factor matrix: 
#>        H1     H2    H3
#> H1  1.000  0.308 0.025
#> H2  3.243  1.000 0.080
#> H3 40.323 12.434 1.000
#> --- 
#> note: equal hypothesis prior probabilities


#####################################
##### example 5: many relations #####
#####################################

# data
Y <- bfi

hypothesis <- c("g1_A1--A2 > g2_A1--A2 & g1_A1--A3 = g2_A1--A3;
                 g1_A1--A2 = g2_A1--A2 & g1_A1--A3 = g2_A1--A3;
                 g1_A1--A2 = g2_A1--A2 = g1_A1--A3 = g2_A1--A3")

Ymale   <- subset(Y, gender == 1,
                  select = -c(education,
                              gender))[,1:5]


# females
Yfemale <- subset(Y, gender == 2,
                     select = -c(education,
                                 gender))[,1:5]

test <- ggm_compare_confirm(Ymale,
                            Yfemale,
                             hypothesis = hypothesis,
                             iter = 250,
                             progress = FALSE)

# print
test
#> BGGM: Bayesian Gaussian Graphical Models 
#> Type: continuous 
#> --- 
#> Posterior Samples: 250 
#>   Group 1: 896 
#>   Group 2: 1813 
#> Variables (p): 5 
#> Relations: 10 
#> Delta: 3 
#> --- 
#> Call:
#> ggm_compare_confirm(Ymale, Yfemale, hypothesis = hypothesis, 
#>     iter = 250, progress = FALSE)
#> --- 
#> Hypotheses: 
#> 
#> H1: g1_A1--A2>g2_A1--A2&g1_A1--A3=g2_A1--A3
#> H2: g1_A1--A2=g2_A1--A2&g1_A1--A3=g2_A1--A3
#> H3: g1_A1--A2=g2_A1--A2=g1_A1--A3=g2_A1--A3
#> H4: complement
#> --- 
#> Posterior prob: 
#> 
#> p(H1|data) = 0.117
#> p(H2|data) = 0.88
#> p(H3|data) = 0.001
#> p(H4|data) = 0.003
#> --- 
#> Bayes factor matrix: 
#>       H1    H2       H3      H4
#> H1 1.000 0.133  218.194  42.427
#> H2 7.527 1.000 1642.242 319.328
#> H3 0.005 0.001    1.000   0.194
#> H4 0.024 0.003    5.143   1.000
#> --- 
#> note: equal hypothesis prior probabilities
# }
```
