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
#>  interested--disinterested    -0.177   0.104  -0.377   0.033
#>        interested--excited     0.378   0.091   0.192   0.554
#>     disinterested--excited    -0.181   0.111  -0.399   0.039
#>          interested--upset    -0.211   0.096  -0.388  -0.017
#>       disinterested--upset    -0.045   0.108  -0.241   0.176
#>             excited--upset    -0.136   0.101  -0.335   0.068
#>         interested--strong     0.329   0.092   0.149   0.508
#>      disinterested--strong     0.110   0.103  -0.085   0.306
#>            excited--strong     0.491   0.076   0.338   0.628
#>              upset--strong     0.119   0.100  -0.076   0.323
#>       interested--stressed     0.269   0.095   0.083   0.450
#>    disinterested--stressed     0.160   0.097  -0.031   0.340
#>          excited--stressed    -0.161   0.107  -0.373   0.062
#>            upset--stressed     0.365   0.085   0.197   0.530
#>           strong--stressed    -0.015   0.105  -0.220   0.190
#>          interested--steps     0.072   0.103  -0.131   0.271
#>       disinterested--steps    -0.093   0.104  -0.295   0.114
#>             excited--steps    -0.015   0.106  -0.217   0.187
#>               upset--steps    -0.044   0.107  -0.236   0.179
#>              strong--steps     0.185   0.098  -0.011   0.367
#>            stressed--steps    -0.010   0.099  -0.207   0.183
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
#>     interested.l1     0.225   0.179  -0.129   0.577
#>  disinterested.l1    -0.048   0.120  -0.284   0.194
#>        excited.l1    -0.084   0.192  -0.465   0.296
#>          upset.l1    -0.154   0.130  -0.411   0.101
#>         strong.l1     0.031   0.172  -0.308   0.373
#>       stressed.l1    -0.017   0.122  -0.256   0.224
#>          steps.l1    -0.155   0.114  -0.378   0.073
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.017   0.183  -0.382   0.351
#>  disinterested.l1    -0.006   0.122  -0.240   0.233
#>        excited.l1    -0.182   0.194  -0.566   0.193
#>          upset.l1     0.257   0.130   0.008   0.515
#>         strong.l1     0.174   0.174  -0.167   0.519
#>       stressed.l1    -0.012   0.123  -0.252   0.228
#>          steps.l1     0.181   0.115  -0.038   0.405
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.180   0.181  -0.179   0.530
#>  disinterested.l1     0.057   0.122  -0.176   0.296
#>        excited.l1     0.003   0.196  -0.381   0.389
#>          upset.l1    -0.096   0.131  -0.355   0.164
#>         strong.l1     0.029   0.176  -0.319   0.367
#>       stressed.l1    -0.029   0.124  -0.271   0.215
#>          steps.l1    -0.209   0.118  -0.442   0.023
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.100   0.172  -0.438   0.241
#>  disinterested.l1    -0.018   0.118  -0.248   0.211
#>        excited.l1     0.052   0.191  -0.316   0.423
#>          upset.l1     0.432   0.123   0.196   0.675
#>         strong.l1     0.048   0.171  -0.288   0.385
#>       stressed.l1    -0.047   0.116  -0.274   0.182
#>          steps.l1     0.152   0.110  -0.065   0.367
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.177   0.181  -0.179   0.523
#>  disinterested.l1     0.052   0.122  -0.194   0.289
#>        excited.l1    -0.083   0.196  -0.466   0.314
#>          upset.l1     0.056   0.132  -0.200   0.313
#>         strong.l1     0.185   0.177  -0.163   0.535
#>       stressed.l1    -0.072   0.124  -0.312   0.173
#>          steps.l1    -0.091   0.118  -0.326   0.145
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.014   0.175  -0.331   0.353
#>  disinterested.l1     0.089   0.118  -0.142   0.317
#>        excited.l1     0.083   0.189  -0.293   0.453
#>          upset.l1     0.319   0.125   0.078   0.572
#>         strong.l1    -0.063   0.171  -0.397   0.270
#>       stressed.l1     0.149   0.116  -0.075   0.377
#>          steps.l1     0.204   0.109  -0.010   0.415
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.107   0.182  -0.240   0.467
#>  disinterested.l1    -0.021   0.126  -0.268   0.226
#>        excited.l1     0.098   0.198  -0.290   0.487
#>          upset.l1    -0.093   0.132  -0.351   0.169
#>         strong.l1    -0.180   0.180  -0.538   0.174
#>       stressed.l1     0.131   0.125  -0.110   0.377
#>          steps.l1     0.041   0.117  -0.190   0.265
#> ---

# }
```
