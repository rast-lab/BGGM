# VAR: Estimation

Estimate VAR(1) models by efficiently sampling from the posterior
distribution. This provides two graphical structures: (1) a network of
undirected relations (the GGM, controlling for the lagged predictors)
and (2) a network of directed relations (the lagged coefficients). Note
that in the graphical modeling literature, this model is also known as a
time series chain graphical model (Abegaz and Wit 2013) .

## Usage

``` r
var_estimate(
  Y,
  rho_sd = sqrt(1/3),
  beta_sd = 1,
  iter = 5000,
  progress = TRUE,
  seed = NULL,
  ...
)
```

## Arguments

- Y:

  Matrix (or data frame) of dimensions *n* (observations) by *p*
  (variables).

- rho_sd:

  Numeric. Scale of the prior distribution for the partial correlations,
  approximately the standard deviation of a beta distribution (defaults
  to sqrt(1/3) as this results to delta = 2, and a uniform distribution
  across the partial correlations).

- beta_sd:

  Numeric. Standard deviation of the prior distribution for the
  regression coefficients (defaults to 1). The prior is by default
  centered at zero and follows a normal distribution (Equation 9, Sinay
  and Hsu 2014)

- iter:

  Number of iterations (posterior samples; defaults to 5000).

- progress:

  Logical. Should a progress bar be included (defaults to `TRUE`) ?

- seed:

  An integer for the random seed (defaults to 1).

- ...:

  Currently ignored.

## Value

An object of class `var_estimate` containing a lot of information that
is used for printing and plotting the results. For users of **BGGM**,
the following are the useful objects:

- `beta_mu` A matrix including the regression coefficients (posterior
  mean).

- `pcor_mu` Partial correlation matrix (posterior mean).

- `fit` A list including the posterior samples.

## Details

Each time series in `Y` is standardized (mean = 0; standard deviation =
1).

## Note

**Regularization**:

A Bayesian ridge regression can be fitted by decreasing `beta_sd` (e.g.,
`beta_sd = 0.25`). This could be advantageous for forecasting
(out-of-sample prediction) in particular.

## References

Abegaz F, Wit E (2013). “Sparse time series chain graphical models for
reconstructing genetic networks.” *Biostatistics*, **14**(3), 586–599.
[doi:10.1093/biostatistics/kxt005](https://doi.org/10.1093/biostatistics/kxt005)
.  
  
Sinay MS, Hsu JS (2014). “Bayesian inference of a multivariate
regression model.” *Journal of Probability and Statistics*, **2014**.

## Examples

``` r
# \donttest{
# data
Y <- subset(ifit, id == 1)[,-1]

# use alias (var_estimate also works)
fit <- var_estimate(Y, progress = FALSE)

fit
#> BGGM: Bayesian Gaussian Graphical Models 
#> --- 
#> Vector Autoregressive Model (VAR) 
#> --- 
#> Posterior Samples: 5000 
#> Observations (n): 94 
#> Nodes (p): 7 
#> --- 
#> Call: 
#> var_estimate(Y = Y, progress = FALSE)
#> --- 
#> Partial Correlations: 
#> 
#>               interested disinterested excited  upset strong stressed  steps
#> interested         0.000        -0.174   0.379 -0.203  0.328    0.265  0.068
#> disinterested     -0.174         0.000  -0.176 -0.042  0.100    0.160 -0.096
#> excited            0.379        -0.176   0.000 -0.137  0.495   -0.158 -0.007
#> upset             -0.203        -0.042  -0.137  0.000  0.126    0.352 -0.063
#> strong             0.328         0.100   0.495  0.126  0.000   -0.021  0.180
#> stressed           0.265         0.160  -0.158  0.352 -0.021    0.000 -0.007
#> steps              0.068        -0.096  -0.007 -0.063  0.180   -0.007  0.000
#> --- 
#> Coefficients: 
#> 
#>                  interested disinterested excited  upset strong stressed  steps
#> interested.l1         0.221        -0.010   0.178 -0.099  0.175    0.012  0.106
#> disinterested.l1     -0.048        -0.004   0.058 -0.017  0.052    0.089 -0.025
#> excited.l1           -0.082        -0.188   0.004  0.054 -0.082    0.082  0.094
#> upset.l1             -0.155         0.256  -0.098  0.430  0.054    0.315 -0.091
#> strong.l1             0.024         0.176   0.025  0.046  0.181   -0.062 -0.178
#> stressed.l1          -0.019        -0.009  -0.032 -0.043 -0.073    0.153  0.131
#> steps.l1             -0.153         0.179  -0.206  0.153 -0.090    0.204  0.042
#> --- 
#> Date: Thu Sep  3 01:24:44 2026 

# }
```
