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
#> interested         0.000        -0.167   0.376 -0.204  0.326    0.284  0.076
#> disinterested     -0.167         0.000  -0.186 -0.028  0.107    0.154 -0.090
#> excited            0.376        -0.186   0.000 -0.128  0.496   -0.178 -0.006
#> upset             -0.204        -0.028  -0.128  0.000  0.103    0.355 -0.049
#> strong             0.326         0.107   0.496  0.103  0.000   -0.005  0.180
#> stressed           0.284         0.154  -0.178  0.355 -0.005    0.000 -0.023
#> steps              0.076        -0.090  -0.006 -0.049  0.180   -0.023  0.000
#> --- 
#> Coefficients: 
#> 
#>                  interested disinterested excited  upset strong stressed  steps
#> interested.l1         0.222        -0.013   0.179 -0.101  0.176    0.013  0.105
#> disinterested.l1     -0.046        -0.007   0.056 -0.020  0.050    0.088 -0.024
#> excited.l1           -0.078        -0.187   0.004  0.053 -0.085    0.087  0.095
#> upset.l1             -0.152         0.256  -0.097  0.428  0.057    0.316 -0.095
#> strong.l1             0.024         0.177   0.022  0.049  0.182   -0.066 -0.180
#> stressed.l1          -0.020        -0.012  -0.029 -0.042 -0.074    0.153  0.133
#> steps.l1             -0.155         0.181  -0.207  0.149 -0.090    0.202  0.042
#> --- 
#> Date: Thu Jul  2 23:56:41 2026 

# }
```
