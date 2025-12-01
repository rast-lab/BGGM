# GGM Compare: Posterior Predictive Check

Compare GGMs with a posterior predicitve check (Gelman et al. 1996) .
This method was introduced in Williams et al. (2020) . Currently, there
is a `global` (the entire GGM) and a `nodewise` test. The default is to
compare GGMs with respect to the posterior predictive distribution of
Kullback Leibler divergence and the sum of squared errors. It is also
possible to compare the GGMs with a user defined test-statistic.

## Usage

``` r
ggm_compare_ppc(
  ...,
  test = "global",
  iter = 5000,
  FUN = NULL,
  custom_obs = NULL,
  loss = TRUE,
  progress = TRUE
)
```

## Arguments

- ...:

  At least two matrices (or data frames) of dimensions *n*
  (observations) by *p* (variables).

- test:

  Which test should be performed (defaults to `"global"`) ? The options
  include `global` and `nodewise`.

- iter:

  Number of replicated datasets used to construct the predictivie
  distribution (defaults to 5000).

- FUN:

  An optional function for comparing GGMs that returns a number. See
  **Details**.

- custom_obs:

  Number corresponding to the observed score for comparing the GGMs.
  This is required if a function is provided in `FUN`. See **Details**.

- loss:

  Logical. If a function is provided, is the measure a "loss function"
  (i.e., a large score is bad thing). This determines how the *p*-value
  is computed. See **Details**.

- progress:

  Logical. Should a progress bar be included (defaults to `TRUE`) ?

## Value

The returned object of class `ggm_compare_ppc` contains a lot of
information that is used for printing and plotting the results. For
users of **BGGM**, the following are the useful objects:

`test = "global"`

- `ppp_jsd` posterior predictive *p*-values (JSD).

- `ppp_sse` posterior predictive *p*-values (SSE).

- `predictive_jsd` list containing the posterior predictive
  distributions (JSD).

- `predictive_sse` list containing the posterior predictive
  distributions (SSE).

- `obs_jsd` list containing the observed error (JSD).

- `obs_sse` list containing the observed error (SSE).

`test = "nodewise"`

- `ppp_jsd` posterior predictive *p*-values (JSD).

- `predictive_jsd` list containing the posterior predictive
  distributions (JSD).

- `obs_jsd` list containing the observed error (JSD).

`FUN = f()`

- `ppp_custom` posterior predictive *p*-values (custom).

- `predictive_custom` posterior predictive distributions (custom).

- `obs_custom` observed error (custom).

## Details

The `FUN` argument allows for a user defined test-statisic (the measure
used to compare the GGMs). The function must include only two agruments,
each of which corresponds to a dataset. For example,
`f <- function(Yg1, Yg2)`, where each Y is dataset of dimensions *n* by
*p*. The groups are then compare within the function, returning a single
number. An example is provided below.

Further, when using a custom function care must be taken when specifying
the argument `loss`. We recommended to visualize the results with `plot`
to ensure the *p*-value was computed in the right direction.

## Note

**Interpretation**:

The primary test-statistic is symmetric KL-divergence that is termed
Jensen-Shannon divergence (JSD). This is in essence a likelihood ratio
that provides the "distance" between two multivariate normal
distributions. The basic idea is to (1) compute the posterior predictive
distribution, assuming group equality (the null model). This provides
the error that we would expect to see under the null model; (2) compute
JSD for the observed groups; and (3) compare the observed JSD to the
posterior predictive distribution, from which a posterior predictive
*p*-value is computed.

For the `global` check, the sum of squared error is also provided. This
is computed from the partial correlation matrices and it is analagous to
the strength test in van Borkulo et al. (2017) . The `nodewise` test
compares the posterior predictive distribution for each node. This is
based on the correspondence between the inverse covariance matrix and
multiple regresssion (Kwan 2014; Stephens 1998) .

If the null model is `not` rejected, note that this does `not` provide
evidence for equality! Further, if the null model is rejected, this
means that the assumption of group equality is not tenable–the groups
are different.

**Alternative Methods**:

There are several methods in **BGGM** for comparing groups. See
[`ggm_compare_estimate`](https://rast-lab.github.io/BGGM/reference/ggm_compare_estimate.md)
(posterior differences for the partial correlations),
[`ggm_compare_explore`](https://rast-lab.github.io/BGGM/reference/ggm_compare_explore.md)
(exploratory hypothesis testing), and
[`ggm_compare_confirm`](https://rast-lab.github.io/BGGM/reference/ggm_compare_confirm.md)
(confirmatory hypothesis testing).

## References

Gelman A, Meng X, Stern H (1996). “Posterior predictive assessment of
model fitness via realized discrepancies.” *Statistica sinica*,
733–760.  
  
Kwan CC (2014). “A regression-based interpretation of the inverse of the
sample covariance matrix.” *Spreadsheets in Education*, **7**(1),
4613.  
  
Stephens G (1998). “On the Inverse of the Covariance Matrix in Portfolio
Analysis.” *The Journal of Finance*, **53**(5), 1821–1827.  
  
van Borkulo CD, Boschloo L, Kossakowski J, Tio P, Schoevers RA, Borsboom
D, Waldorp LJ (2017). “Comparing network structures on three aspects: A
permutation test.” *Manuscript submitted for publication*.  
  
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

#############################
######### global ############
#############################


# males
Ym <- subset(Y, gender == 1,
             select = - c(gender, education))

# females

Yf <- subset(Y, gender == 2,
             select = - c(gender, education))


global_test <- ggm_compare_ppc(Ym, Yf,
                               iter = 250)
#> BGGM: Predictive Check (Contrast 1)
#> BGGM: Finished

global_test
#> BGGM: Bayesian Gaussian Graphical Models 
#> --- 
#> Test: Global Predictive Check 
#> Posterior Samples: 250 
#>   Group 1: 805 
#>   Group 2: 1631 
#> Nodes:  25 
#> Relations: 300 
#> --- 
#> Call: 
#> ggm_compare_ppc(Ym, Yf, iter = 250)
#> --- 
#> Symmetric KL divergence (JSD): 
#>  
#>    contrast JSD.obs p_value
#>  Yg1 vs Yg2   0.442       0
#> --- 
#>  
#> Sum of Squared Error: 
#>  
#>    contrast SSE.obs p.value
#>  Yg1 vs Yg2   0.759       0
#> --- 
#> note:
#> JSD is Jensen-Shannon divergence 


#############################
###### custom function ######
#############################
# example 1

# maximum difference van Borkulo et al. (2017)

f <- function(Yg1, Yg2){

# remove NA
x <- na.omit(Yg1)
y <- na.omit(Yg2)

# nodes
p <- ncol(Yg1)

# identity matrix
I_p <- diag(p)

# partial correlations

pcor_1 <- -(cov2cor(solve(cor(x))) - I_p)
pcor_2 <- -(cov2cor(solve(cor(y))) - I_p)

# max difference
max(abs((pcor_1[upper.tri(I_p)] - pcor_2[upper.tri(I_p)])))

}

# observed difference
obs <- f(Ym, Yf)

global_max <- ggm_compare_ppc(Ym, Yf,
                              iter = 250,
                              FUN = f,
                              custom_obs = obs,
                              progress = FALSE)

global_max
#> BGGM: Bayesian Gaussian Graphical Models 
#> --- 
#> Test: Global Predictive Check 
#> Posterior Samples: 250 
#>   Group 1: 805 
#>   Group 2: 1631 
#> Nodes:  25 
#> Relations: 300 
#> --- 
#> Call: 
#> ggm_compare_ppc(Ym, Yf, iter = 250, FUN = f, custom_obs = obs, 
#>     progress = FALSE)
#> --- 
#> Custom: 
#>  
#>    contrast custom.obs p.value
#>  Yg1 vs Yg2       0.16   0.056
#> --- 


# example 2
# Hamming distance (squared error for adjacency)

f <- function(Yg1, Yg2){

# remove NA
x <- na.omit(Yg1)
y <- na.omit(Yg2)

# nodes
p <- ncol(x)

# identity matrix
I_p <- diag(p)

fit1 <-  estimate(x, analytic = TRUE)
fit2 <-  estimate(y, analytic = TRUE)

sel1 <- select(fit1)
sel2 <- select(fit2)

sum((sel1$adj[upper.tri(I_p)] - sel2$adj[upper.tri(I_p)])^2)

}

# observed difference
obs <- f(Ym, Yf)

global_hd <- ggm_compare_ppc(Ym, Yf,
                            iter = 250,
                            FUN = f,
                            custom_obs  = obs,
                            progress = FALSE)

global_hd
#> BGGM: Bayesian Gaussian Graphical Models 
#> --- 
#> Test: Global Predictive Check 
#> Posterior Samples: 250 
#>   Group 1: 805 
#>   Group 2: 1631 
#> Nodes:  25 
#> Relations: 300 
#> --- 
#> Call: 
#> ggm_compare_ppc(Ym, Yf, iter = 250, FUN = f, custom_obs = obs, 
#>     progress = FALSE)
#> --- 
#> Custom: 
#>  
#>    contrast custom.obs p.value
#>  Yg1 vs Yg2         75   0.528
#> --- 


#############################
########  nodewise ##########
#############################

nodewise <- ggm_compare_ppc(Ym, Yf, iter = 250,
                           test = "nodewise")
#> BGGM: Predictive Check (Contrast 1)
#> BGGM: Finished

nodewise
#> BGGM: Bayesian Gaussian Graphical Models 
#> --- 
#> Test: Nodewise Predictive Check 
#> Posterior Samples: 250 
#>   Group 1: 805 
#>   Group 2: 1631 
#> Nodes:  25 
#> Relations: 300 
#> --- 
#> Call: 
#> ggm_compare_ppc(Ym, Yf, test = "nodewise", iter = 250)
#> --- 
#> Symmetric KL divergence (JSD): 
#>  
#> Yg1 vs Yg2 
#>  Node JSD.obs p_value
#>     1   0.001   0.648
#>     2   0.000   0.944
#>     3   0.006   0.076
#>     4   0.005   0.268
#>     5   0.006   0.104
#>     6   0.002   0.388
#>     7   0.000   0.788
#>     8   0.000   0.800
#>     9   0.000   0.908
#>    10   0.003   0.220
#>    11   0.020   0.012
#>    12   0.009   0.020
#>    13   0.003   0.288
#>    14   0.004   0.120
#>    15   0.011   0.028
#>    16   0.001   0.420
#>    17   0.000   0.620
#>    18   0.000   0.596
#>    19   0.000   0.764
#>    20   0.000   0.656
#>    21   0.000   0.932
#>    22   0.003   0.436
#>    23   0.002   0.352
#>    24   0.013   0.188
#>    25   0.018   0.080
#> --- 
#> 
#> note:
#> JSD is Jensen-Shannon divergence 

# }
```
