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
#> interested         0.000        -0.167   0.394 -0.211  0.322    0.273  0.074
#> disinterested     -0.167         0.000  -0.195 -0.037  0.120    0.155 -0.092
#> excited            0.394        -0.195   0.000 -0.118  0.489   -0.163 -0.014
#> upset             -0.211        -0.037  -0.118  0.000  0.110    0.353 -0.042
#> strong             0.322         0.120   0.489  0.110  0.000   -0.019  0.180
#> stressed           0.273         0.155  -0.163  0.353 -0.019    0.000 -0.017
#> steps              0.074        -0.092  -0.014 -0.042  0.180   -0.017  0.000
#> --- 
#> Coefficients: 
#> 
#>                  interested disinterested excited  upset strong stressed  steps
#> interested.l1         0.224        -0.015   0.182 -0.097  0.177    0.015  0.113
#> disinterested.l1     -0.051        -0.004   0.057 -0.017  0.049    0.088 -0.023
#> excited.l1           -0.082        -0.180   0.003  0.052 -0.084    0.088  0.098
#> upset.l1             -0.152         0.257  -0.095  0.427  0.056    0.317 -0.092
#> strong.l1             0.025         0.171   0.026  0.045  0.182   -0.072 -0.186
#> stressed.l1          -0.020        -0.011  -0.031 -0.045 -0.074    0.152  0.131
#> steps.l1             -0.152         0.182  -0.207  0.152 -0.089    0.206  0.042
#> --- 
#> Date: Tue Dec  2 02:18:26 2025 

# }
```
