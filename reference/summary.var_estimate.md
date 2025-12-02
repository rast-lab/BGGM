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
#>  interested--disinterested    -0.177   0.101  -0.371   0.029
#>        interested--excited     0.382   0.090   0.190   0.545
#>     disinterested--excited    -0.190   0.099  -0.370   0.010
#>          interested--upset    -0.205   0.100  -0.394  -0.008
#>       disinterested--upset    -0.028   0.110  -0.237   0.186
#>             excited--upset    -0.133   0.106  -0.346   0.072
#>         interested--strong     0.324   0.089   0.137   0.491
#>      disinterested--strong     0.114   0.103  -0.103   0.306
#>            excited--strong     0.494   0.078   0.331   0.637
#>              upset--strong     0.117   0.113  -0.105   0.328
#>       interested--stressed     0.271   0.106   0.059   0.470
#>    disinterested--stressed     0.147   0.106  -0.064   0.340
#>          excited--stressed    -0.174   0.107  -0.373   0.056
#>            upset--stressed     0.354   0.088   0.166   0.517
#>           strong--stressed    -0.010   0.107  -0.219   0.202
#>          interested--steps     0.077   0.105  -0.123   0.274
#>       disinterested--steps    -0.082   0.107  -0.288   0.125
#>             excited--steps    -0.014   0.108  -0.227   0.195
#>               upset--steps    -0.050   0.108  -0.275   0.151
#>              strong--steps     0.186   0.101  -0.011   0.379
#>            stressed--steps    -0.020   0.104  -0.227   0.181
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
#>     interested.l1     0.224   0.179  -0.125   0.573
#>  disinterested.l1    -0.052   0.123  -0.291   0.192
#>        excited.l1    -0.083   0.198  -0.477   0.307
#>          upset.l1    -0.150   0.129  -0.403   0.109
#>         strong.l1     0.023   0.176  -0.319   0.377
#>       stressed.l1    -0.022   0.121  -0.257   0.214
#>          steps.l1    -0.156   0.114  -0.377   0.076
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.014   0.179  -0.379   0.334
#>  disinterested.l1    -0.002   0.124  -0.243   0.243
#>        excited.l1    -0.184   0.198  -0.579   0.207
#>          upset.l1     0.255   0.127  -0.001   0.503
#>         strong.l1     0.175   0.177  -0.170   0.518
#>       stressed.l1    -0.009   0.121  -0.247   0.223
#>          steps.l1     0.181   0.111  -0.042   0.395
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.180   0.184  -0.184   0.542
#>  disinterested.l1     0.054   0.125  -0.192   0.300
#>        excited.l1     0.003   0.199  -0.388   0.393
#>          upset.l1    -0.092   0.130  -0.344   0.165
#>         strong.l1     0.026   0.180  -0.325   0.393
#>       stressed.l1    -0.033   0.123  -0.272   0.209
#>          steps.l1    -0.211   0.115  -0.435   0.019
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.102   0.172  -0.428   0.235
#>  disinterested.l1    -0.016   0.118  -0.254   0.212
#>        excited.l1     0.053   0.188  -0.325   0.420
#>          upset.l1     0.428   0.125   0.176   0.673
#>         strong.l1     0.052   0.171  -0.288   0.385
#>       stressed.l1    -0.045   0.118  -0.280   0.186
#>          steps.l1     0.152   0.107  -0.063   0.365
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.174   0.182  -0.188   0.532
#>  disinterested.l1     0.047   0.124  -0.199   0.295
#>        excited.l1    -0.085   0.200  -0.485   0.308
#>          upset.l1     0.060   0.129  -0.190   0.318
#>         strong.l1     0.184   0.179  -0.170   0.540
#>       stressed.l1    -0.077   0.122  -0.324   0.161
#>          steps.l1    -0.096   0.116  -0.316   0.132
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.013   0.172  -0.323   0.354
#>  disinterested.l1     0.090   0.117  -0.146   0.317
#>        excited.l1     0.084   0.187  -0.279   0.442
#>          upset.l1     0.316   0.124   0.079   0.553
#>         strong.l1    -0.065   0.169  -0.396   0.274
#>       stressed.l1     0.151   0.116  -0.073   0.378
#>          steps.l1     0.207   0.108  -0.004   0.417
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.109   0.186  -0.261   0.474
#>  disinterested.l1    -0.024   0.124  -0.264   0.225
#>        excited.l1     0.105   0.203  -0.291   0.524
#>          upset.l1    -0.090   0.132  -0.344   0.175
#>         strong.l1    -0.189   0.181  -0.539   0.169
#>       stressed.l1     0.130   0.128  -0.121   0.376
#>          steps.l1     0.041   0.116  -0.186   0.269
#> ---

# }
```
