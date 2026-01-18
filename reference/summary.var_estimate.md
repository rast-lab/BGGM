# Summary Method for `var_estimate` Objects

Summarize the posterior distribution of each partial correlation and
regression coefficient with the posterior mean, standard deviation, and
credible intervals.

## Usage

``` r
# S3 method for class 'var_estimate'
summary(object, cred = 0.95, ...)
```

## Arguments

- object:

  An object of class `var_estimate`

- cred:

  Numeric. The credible interval width for summarizing the posterior
  distributions (defaults to 0.95; must be between 0 and 1).

- ...:

  Currently ignored.

## Value

A dataframe containing the summarized posterior distributions, including
both the partial correlations and the regression coefficients.

- `pcor_results` A data frame including the summarized partial
  correlations

- `beta_results` A list containing the summarized regression
  coefficients (one data frame for each outcome)

## See also

[`var_estimate`](https://rast-lab.github.io/BGGM/reference/var_estimate.md)

## Examples

``` r
# \donttest{
# data
Y <- subset(ifit, id == 1)[,-1]

# fit model with alias (var_estimate also works)
fit <- var_estimate(Y, progress = FALSE)

# summary ('pcor')
print(
summary(fit, cred = 0.95),
param = "pcor",
)
#> BGGM: Bayesian Gaussian Graphical Models 
#> --- 
#> Vector Autoregressive Model (VAR) 
#> --- 
#> Partial Correlations: 
#> 
#>                   Relation Post.mean Post.sd Cred.lb Cred.ub
#>  interested--disinterested    -0.180   0.096  -0.356   0.014
#>        interested--excited     0.385   0.086   0.208   0.544
#>     disinterested--excited    -0.171   0.100  -0.362   0.026
#>          interested--upset    -0.212   0.102  -0.394  -0.004
#>       disinterested--upset    -0.041   0.103  -0.242   0.165
#>             excited--upset    -0.123   0.102  -0.317   0.084
#>         interested--strong     0.327   0.094   0.139   0.498
#>      disinterested--strong     0.108   0.104  -0.095   0.318
#>            excited--strong     0.493   0.080   0.328   0.640
#>              upset--strong     0.112   0.099  -0.070   0.307
#>       interested--stressed     0.276   0.097   0.078   0.459
#>    disinterested--stressed     0.160   0.107  -0.067   0.356
#>          excited--stressed    -0.177   0.104  -0.377   0.031
#>            upset--stressed     0.356   0.089   0.178   0.535
#>           strong--stressed    -0.009   0.109  -0.236   0.191
#>          interested--steps     0.086   0.102  -0.112   0.285
#>       disinterested--steps    -0.082   0.103  -0.289   0.119
#>             excited--steps    -0.012   0.108  -0.221   0.208
#>               upset--steps    -0.045   0.103  -0.236   0.164
#>              strong--steps     0.172   0.107  -0.052   0.370
#>            stressed--steps    -0.024   0.106  -0.215   0.198
#> --- 
#> 


# summary ('beta')
print(
summary(fit, cred = 0.95),
param = "beta",
)
#> BGGM: Bayesian Gaussian Graphical Models 
#> --- 
#> Vector Autoregressive Model (VAR) 
#> --- 
#> Coefficients: 
#> 
#> interested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.216   0.180  -0.135   0.571
#>  disinterested.l1    -0.049   0.125  -0.296   0.193
#>        excited.l1    -0.079   0.200  -0.475   0.317
#>          upset.l1    -0.157   0.128  -0.403   0.093
#>         strong.l1     0.028   0.176  -0.308   0.381
#>       stressed.l1    -0.018   0.121  -0.255   0.222
#>          steps.l1    -0.155   0.114  -0.376   0.067
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.014   0.178  -0.368   0.337
#>  disinterested.l1    -0.006   0.122  -0.244   0.231
#>        excited.l1    -0.186   0.195  -0.571   0.194
#>          upset.l1     0.259   0.128   0.006   0.521
#>         strong.l1     0.172   0.176  -0.176   0.505
#>       stressed.l1    -0.011   0.119  -0.248   0.217
#>          steps.l1     0.182   0.113  -0.045   0.406
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.176   0.181  -0.175   0.542
#>  disinterested.l1     0.056   0.126  -0.198   0.305
#>        excited.l1     0.005   0.201  -0.390   0.392
#>          upset.l1    -0.100   0.131  -0.355   0.153
#>         strong.l1     0.027   0.180  -0.321   0.373
#>       stressed.l1    -0.030   0.122  -0.261   0.212
#>          steps.l1    -0.210   0.116  -0.432   0.021
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.096   0.172  -0.433   0.233
#>  disinterested.l1    -0.020   0.118  -0.249   0.210
#>        excited.l1     0.052   0.188  -0.321   0.414
#>          upset.l1     0.432   0.124   0.187   0.674
#>         strong.l1     0.045   0.168  -0.284   0.375
#>       stressed.l1    -0.045   0.117  -0.281   0.182
#>          steps.l1     0.149   0.108  -0.063   0.361
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.171   0.183  -0.195   0.538
#>  disinterested.l1     0.050   0.125  -0.188   0.297
#>        excited.l1    -0.082   0.203  -0.477   0.323
#>          upset.l1     0.054   0.131  -0.204   0.314
#>         strong.l1     0.185   0.182  -0.169   0.544
#>       stressed.l1    -0.074   0.124  -0.316   0.168
#>          steps.l1    -0.092   0.116  -0.321   0.139
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.015   0.175  -0.326   0.363
#>  disinterested.l1     0.090   0.119  -0.142   0.324
#>        excited.l1     0.091   0.190  -0.272   0.468
#>          upset.l1     0.320   0.126   0.067   0.569
#>         strong.l1    -0.071   0.170  -0.396   0.265
#>       stressed.l1     0.151   0.117  -0.077   0.379
#>          steps.l1     0.205   0.107  -0.003   0.417
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.105   0.185  -0.252   0.470
#>  disinterested.l1    -0.022   0.126  -0.272   0.220
#>        excited.l1     0.100   0.200  -0.288   0.501
#>          upset.l1    -0.093   0.134  -0.355   0.169
#>         strong.l1    -0.179   0.177  -0.524   0.163
#>       stressed.l1     0.130   0.125  -0.115   0.378
#>          steps.l1     0.040   0.116  -0.190   0.265
#> ---

# }
```
