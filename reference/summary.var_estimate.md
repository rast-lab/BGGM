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
#>  interested--disinterested    -0.172   0.101  -0.364   0.025
#>        interested--excited     0.388   0.089   0.204   0.549
#>     disinterested--excited    -0.175   0.103  -0.365   0.033
#>          interested--upset    -0.201   0.102  -0.387   0.024
#>       disinterested--upset    -0.025   0.105  -0.233   0.177
#>             excited--upset    -0.125   0.103  -0.319   0.087
#>         interested--strong     0.324   0.091   0.131   0.489
#>      disinterested--strong     0.104   0.102  -0.096   0.303
#>            excited--strong     0.494   0.082   0.320   0.634
#>              upset--strong     0.112   0.092  -0.073   0.292
#>       interested--stressed     0.269   0.099   0.069   0.453
#>    disinterested--stressed     0.150   0.104  -0.059   0.343
#>          excited--stressed    -0.172   0.102  -0.371   0.033
#>            upset--stressed     0.355   0.088   0.188   0.529
#>           strong--stressed    -0.006   0.106  -0.211   0.203
#>          interested--steps     0.075   0.104  -0.135   0.276
#>       disinterested--steps    -0.078   0.106  -0.283   0.139
#>             excited--steps    -0.010   0.101  -0.201   0.190
#>               upset--steps    -0.047   0.104  -0.250   0.151
#>              strong--steps     0.185   0.102  -0.026   0.379
#>            stressed--steps    -0.020   0.108  -0.231   0.198
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
#>     interested.l1     0.226   0.179  -0.128   0.585
#>  disinterested.l1    -0.051   0.122  -0.285   0.188
#>        excited.l1    -0.086   0.196  -0.474   0.311
#>          upset.l1    -0.152   0.130  -0.403   0.101
#>         strong.l1     0.026   0.176  -0.333   0.363
#>       stressed.l1    -0.022   0.122  -0.261   0.221
#>          steps.l1    -0.155   0.113  -0.378   0.066
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.016   0.177  -0.367   0.340
#>  disinterested.l1    -0.002   0.121  -0.241   0.231
#>        excited.l1    -0.179   0.197  -0.561   0.201
#>          upset.l1     0.258   0.128   0.005   0.499
#>         strong.l1     0.174   0.178  -0.177   0.519
#>       stressed.l1    -0.009   0.120  -0.243   0.222
#>          steps.l1     0.183   0.112  -0.034   0.399
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.182   0.185  -0.186   0.545
#>  disinterested.l1     0.054   0.122  -0.178   0.295
#>        excited.l1     0.002   0.199  -0.382   0.398
#>          upset.l1    -0.095   0.132  -0.350   0.165
#>         strong.l1     0.024   0.181  -0.345   0.367
#>       stressed.l1    -0.034   0.124  -0.275   0.208
#>          steps.l1    -0.208   0.116  -0.435   0.021
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.097   0.174  -0.442   0.239
#>  disinterested.l1    -0.017   0.116  -0.250   0.209
#>        excited.l1     0.054   0.186  -0.308   0.417
#>          upset.l1     0.429   0.123   0.192   0.670
#>         strong.l1     0.046   0.167  -0.280   0.371
#>       stressed.l1    -0.044   0.115  -0.270   0.181
#>          steps.l1     0.150   0.109  -0.061   0.363
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.179   0.184  -0.182   0.534
#>  disinterested.l1     0.050   0.125  -0.191   0.296
#>        excited.l1    -0.084   0.203  -0.479   0.317
#>          upset.l1     0.059   0.132  -0.197   0.320
#>         strong.l1     0.180   0.181  -0.188   0.538
#>       stressed.l1    -0.076   0.125  -0.320   0.173
#>          steps.l1    -0.091   0.116  -0.319   0.137
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.012   0.168  -0.316   0.345
#>  disinterested.l1     0.090   0.116  -0.144   0.313
#>        excited.l1     0.088   0.184  -0.276   0.451
#>          upset.l1     0.316   0.123   0.072   0.558
#>         strong.l1    -0.065   0.165  -0.389   0.261
#>       stressed.l1     0.153   0.114  -0.073   0.371
#>          steps.l1     0.203   0.107  -0.005   0.415
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.109   0.185  -0.267   0.469
#>  disinterested.l1    -0.023   0.125  -0.275   0.227
#>        excited.l1     0.101   0.206  -0.306   0.506
#>          upset.l1    -0.088   0.131  -0.353   0.169
#>         strong.l1    -0.185   0.184  -0.539   0.172
#>       stressed.l1     0.129   0.124  -0.111   0.375
#>          steps.l1     0.042   0.115  -0.183   0.274
#> ---

# }
```
