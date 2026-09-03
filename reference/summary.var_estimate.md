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
#>  interested--disinterested    -0.190   0.102  -0.379   0.020
#>        interested--excited     0.378   0.093   0.186   0.545
#>     disinterested--excited    -0.172   0.105  -0.370   0.039
#>          interested--upset    -0.212   0.099  -0.400  -0.017
#>       disinterested--upset    -0.030   0.103  -0.240   0.167
#>             excited--upset    -0.118   0.101  -0.318   0.075
#>         interested--strong     0.334   0.096   0.141   0.511
#>      disinterested--strong     0.104   0.107  -0.099   0.309
#>            excited--strong     0.492   0.076   0.330   0.629
#>              upset--strong     0.115   0.102  -0.087   0.311
#>       interested--stressed     0.280   0.103   0.073   0.472
#>    disinterested--stressed     0.164   0.102  -0.036   0.369
#>          excited--stressed    -0.164   0.111  -0.372   0.063
#>            upset--stressed     0.349   0.088   0.165   0.519
#>           strong--stressed    -0.020   0.112  -0.226   0.211
#>          interested--steps     0.081   0.102  -0.116   0.281
#>       disinterested--steps    -0.075   0.109  -0.291   0.134
#>             excited--steps    -0.003   0.100  -0.199   0.181
#>               upset--steps    -0.053   0.100  -0.238   0.152
#>              strong--steps     0.171   0.099  -0.023   0.353
#>            stressed--steps    -0.024   0.098  -0.223   0.165
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
#>     interested.l1     0.218   0.183  -0.144   0.577
#>  disinterested.l1    -0.049   0.124  -0.299   0.196
#>        excited.l1    -0.078   0.201  -0.471   0.324
#>          upset.l1    -0.154   0.129  -0.403   0.099
#>         strong.l1     0.028   0.181  -0.332   0.383
#>       stressed.l1    -0.020   0.124  -0.263   0.229
#>          steps.l1    -0.156   0.113  -0.376   0.067
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.016   0.175  -0.364   0.326
#>  disinterested.l1    -0.004   0.122  -0.240   0.234
#>        excited.l1    -0.181   0.196  -0.563   0.213
#>          upset.l1     0.255   0.129   0.007   0.506
#>         strong.l1     0.173   0.174  -0.170   0.514
#>       stressed.l1    -0.009   0.121  -0.244   0.230
#>          steps.l1     0.182   0.113  -0.037   0.406
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.175   0.185  -0.193   0.539
#>  disinterested.l1     0.056   0.124  -0.188   0.304
#>        excited.l1     0.006   0.201  -0.387   0.402
#>          upset.l1    -0.097   0.129  -0.354   0.155
#>         strong.l1     0.029   0.178  -0.313   0.376
#>       stressed.l1    -0.032   0.124  -0.283   0.215
#>          steps.l1    -0.209   0.115  -0.440   0.014
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.101   0.172  -0.438   0.238
#>  disinterested.l1    -0.020   0.117  -0.253   0.209
#>        excited.l1     0.050   0.189  -0.314   0.424
#>          upset.l1     0.429   0.122   0.193   0.676
#>         strong.l1     0.049   0.171  -0.284   0.377
#>       stressed.l1    -0.044   0.115  -0.266   0.189
#>          steps.l1     0.151   0.109  -0.061   0.366
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.171   0.185  -0.195   0.533
#>  disinterested.l1     0.050   0.127  -0.200   0.296
#>        excited.l1    -0.081   0.205  -0.497   0.323
#>          upset.l1     0.054   0.131  -0.196   0.312
#>         strong.l1     0.185   0.180  -0.163   0.542
#>       stressed.l1    -0.073   0.127  -0.327   0.177
#>          steps.l1    -0.090   0.118  -0.322   0.145
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.008   0.170  -0.329   0.331
#>  disinterested.l1     0.089   0.117  -0.140   0.317
#>        excited.l1     0.088   0.188  -0.277   0.456
#>          upset.l1     0.313   0.124   0.066   0.550
#>         strong.l1    -0.065   0.170  -0.402   0.273
#>       stressed.l1     0.152   0.115  -0.075   0.376
#>          steps.l1     0.204   0.107  -0.006   0.413
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.105   0.183  -0.250   0.472
#>  disinterested.l1    -0.024   0.125  -0.267   0.220
#>        excited.l1     0.103   0.201  -0.287   0.491
#>          upset.l1    -0.092   0.129  -0.342   0.162
#>         strong.l1    -0.181   0.180  -0.547   0.173
#>       stressed.l1     0.132   0.125  -0.106   0.374
#>          steps.l1     0.040   0.115  -0.191   0.265
#> ---

# }
```
