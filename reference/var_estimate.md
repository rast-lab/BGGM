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
#> interested         0.000        -0.163   0.386 -0.214  0.323    0.273  0.089
#> disinterested     -0.163         0.000  -0.177 -0.038  0.088    0.163 -0.098
#> excited            0.386        -0.177   0.000 -0.108  0.493   -0.173 -0.015
#> upset             -0.214        -0.038  -0.108  0.000  0.106    0.358 -0.043
#> strong             0.323         0.088   0.493  0.106  0.000   -0.001  0.172
#> stressed           0.273         0.163  -0.173  0.358 -0.001    0.000 -0.017
#> steps              0.089        -0.098  -0.015 -0.043  0.172   -0.017  0.000
#> --- 
#> Coefficients: 
#> 
#>                  interested disinterested excited  upset strong stressed  steps
#> interested.l1         0.221        -0.011   0.177 -0.098  0.174    0.013  0.104
#> disinterested.l1     -0.052        -0.002   0.054 -0.015  0.049    0.091 -0.027
#> excited.l1           -0.084        -0.184   0.004  0.057 -0.083    0.089  0.099
#> upset.l1             -0.152         0.256  -0.093  0.427  0.057    0.312 -0.091
#> strong.l1             0.030         0.170   0.029  0.044  0.184   -0.067 -0.182
#> stressed.l1          -0.017        -0.010  -0.029 -0.042 -0.074    0.154  0.130
#> steps.l1             -0.155         0.181  -0.206  0.153 -0.090    0.204  0.041
#> --- 
#> Date: Fri Jan 16 23:20:32 2026 

# }
```
