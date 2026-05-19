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
#>  interested--disinterested    -0.180   0.098  -0.371   0.002
#>        interested--excited     0.386   0.084   0.207   0.539
#>     disinterested--excited    -0.168   0.106  -0.374   0.046
#>          interested--upset    -0.216   0.103  -0.410  -0.019
#>       disinterested--upset    -0.042   0.101  -0.255   0.152
#>             excited--upset    -0.115   0.102  -0.305   0.095
#>         interested--strong     0.330   0.094   0.146   0.505
#>      disinterested--strong     0.091   0.100  -0.102   0.285
#>            excited--strong     0.483   0.080   0.312   0.627
#>              upset--strong     0.100   0.105  -0.104   0.303
#>       interested--stressed     0.279   0.094   0.088   0.457
#>    disinterested--stressed     0.166   0.104  -0.050   0.359
#>          excited--stressed    -0.163   0.102  -0.362   0.040
#>            upset--stressed     0.353   0.088   0.171   0.510
#>           strong--stressed    -0.022   0.106  -0.229   0.187
#>          interested--steps     0.067   0.109  -0.152   0.274
#>       disinterested--steps    -0.091   0.103  -0.288   0.111
#>             excited--steps    -0.005   0.101  -0.204   0.192
#>               upset--steps    -0.045   0.101  -0.232   0.167
#>              strong--steps     0.186   0.107  -0.019   0.405
#>            stressed--steps    -0.015   0.106  -0.215   0.200
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
#>     interested.l1     0.222   0.179  -0.124   0.574
#>  disinterested.l1    -0.047   0.123  -0.288   0.192
#>        excited.l1    -0.080   0.199  -0.478   0.317
#>          upset.l1    -0.151   0.130  -0.404   0.110
#>         strong.l1     0.025   0.180  -0.325   0.370
#>       stressed.l1    -0.021   0.121  -0.252   0.219
#>          steps.l1    -0.152   0.112  -0.373   0.068
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.017   0.179  -0.369   0.333
#>  disinterested.l1    -0.006   0.122  -0.247   0.233
#>        excited.l1    -0.183   0.194  -0.560   0.202
#>          upset.l1     0.256   0.128   0.008   0.504
#>         strong.l1     0.174   0.176  -0.166   0.525
#>       stressed.l1    -0.010   0.118  -0.244   0.219
#>          steps.l1     0.181   0.110  -0.036   0.399
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.180   0.182  -0.165   0.541
#>  disinterested.l1     0.058   0.127  -0.196   0.304
#>        excited.l1     0.004   0.198  -0.381   0.389
#>          upset.l1    -0.094   0.130  -0.350   0.163
#>         strong.l1     0.025   0.181  -0.322   0.378
#>       stressed.l1    -0.034   0.122  -0.268   0.205
#>          steps.l1    -0.206   0.114  -0.434   0.012
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.098   0.172  -0.434   0.238
#>  disinterested.l1    -0.021   0.117  -0.258   0.203
#>        excited.l1     0.049   0.186  -0.322   0.414
#>          upset.l1     0.425   0.122   0.183   0.658
#>         strong.l1     0.048   0.167  -0.282   0.377
#>       stressed.l1    -0.039   0.117  -0.269   0.191
#>          steps.l1     0.150   0.108  -0.066   0.363
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.173   0.184  -0.185   0.542
#>  disinterested.l1     0.054   0.127  -0.205   0.301
#>        excited.l1    -0.082   0.200  -0.471   0.314
#>          upset.l1     0.057   0.134  -0.211   0.319
#>         strong.l1     0.182   0.182  -0.182   0.546
#>       stressed.l1    -0.076   0.124  -0.315   0.167
#>          steps.l1    -0.089   0.116  -0.324   0.136
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.011   0.170  -0.320   0.348
#>  disinterested.l1     0.088   0.117  -0.144   0.318
#>        excited.l1     0.086   0.187  -0.285   0.460
#>          upset.l1     0.315   0.124   0.072   0.552
#>         strong.l1    -0.064   0.166  -0.386   0.266
#>       stressed.l1     0.155   0.117  -0.077   0.391
#>          steps.l1     0.205   0.109  -0.003   0.419
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.106   0.183  -0.259   0.468
#>  disinterested.l1    -0.020   0.125  -0.270   0.224
#>        excited.l1     0.105   0.201  -0.292   0.497
#>          upset.l1    -0.091   0.132  -0.353   0.168
#>         strong.l1    -0.186   0.182  -0.551   0.165
#>       stressed.l1     0.132   0.125  -0.111   0.378
#>          steps.l1     0.043   0.116  -0.185   0.269
#> ---

# }
```
