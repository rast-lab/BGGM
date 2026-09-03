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
#>  interested--disinterested    -0.182   0.104  -0.383   0.027
#>        interested--excited     0.388   0.089   0.198   0.556
#>     disinterested--excited    -0.179   0.105  -0.367   0.033
#>          interested--upset    -0.211   0.103  -0.392  -0.001
#>       disinterested--upset    -0.049   0.105  -0.253   0.161
#>             excited--upset    -0.130   0.105  -0.334   0.081
#>         interested--strong     0.326   0.091   0.140   0.493
#>      disinterested--strong     0.109   0.102  -0.093   0.304
#>            excited--strong     0.485   0.076   0.328   0.625
#>              upset--strong     0.112   0.109  -0.106   0.317
#>       interested--stressed     0.288   0.092   0.102   0.460
#>    disinterested--stressed     0.170   0.099  -0.029   0.361
#>          excited--stressed    -0.167   0.095  -0.349   0.028
#>            upset--stressed     0.356   0.089   0.165   0.525
#>           strong--stressed    -0.020   0.101  -0.220   0.169
#>          interested--steps     0.079   0.106  -0.128   0.280
#>       disinterested--steps    -0.089   0.102  -0.285   0.118
#>             excited--steps    -0.009   0.101  -0.202   0.195
#>               upset--steps    -0.046   0.110  -0.250   0.179
#>              strong--steps     0.182   0.100  -0.012   0.372
#>            stressed--steps    -0.014   0.104  -0.216   0.196
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
#>     interested.l1     0.223   0.177  -0.126   0.563
#>  disinterested.l1    -0.048   0.125  -0.291   0.197
#>        excited.l1    -0.084   0.197  -0.470   0.313
#>          upset.l1    -0.153   0.130  -0.405   0.108
#>         strong.l1     0.031   0.174  -0.313   0.378
#>       stressed.l1    -0.022   0.123  -0.265   0.217
#>          steps.l1    -0.157   0.112  -0.380   0.066
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.013   0.178  -0.365   0.336
#>  disinterested.l1    -0.004   0.121  -0.236   0.237
#>        excited.l1    -0.179   0.195  -0.559   0.203
#>          upset.l1     0.259   0.129   0.005   0.509
#>         strong.l1     0.169   0.177  -0.181   0.514
#>       stressed.l1    -0.011   0.121  -0.244   0.225
#>          steps.l1     0.183   0.113  -0.036   0.401
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.177   0.180  -0.185   0.531
#>  disinterested.l1     0.055   0.124  -0.187   0.296
#>        excited.l1     0.004   0.203  -0.395   0.397
#>          upset.l1    -0.096   0.131  -0.355   0.167
#>         strong.l1     0.029   0.179  -0.321   0.387
#>       stressed.l1    -0.033   0.125  -0.278   0.214
#>          steps.l1    -0.210   0.116  -0.436   0.015
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.099   0.170  -0.430   0.240
#>  disinterested.l1    -0.017   0.118  -0.244   0.220
#>        excited.l1     0.057   0.189  -0.318   0.433
#>          upset.l1     0.430   0.124   0.191   0.677
#>         strong.l1     0.042   0.166  -0.289   0.366
#>       stressed.l1    -0.043   0.117  -0.267   0.186
#>          steps.l1     0.153   0.108  -0.055   0.362
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.173   0.184  -0.189   0.541
#>  disinterested.l1     0.049   0.126  -0.186   0.304
#>        excited.l1    -0.082   0.204  -0.480   0.323
#>          upset.l1     0.057   0.132  -0.206   0.321
#>         strong.l1     0.185   0.180  -0.165   0.542
#>       stressed.l1    -0.076   0.124  -0.319   0.163
#>          steps.l1    -0.094   0.117  -0.320   0.139
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.011   0.166  -0.313   0.334
#>  disinterested.l1     0.090   0.115  -0.136   0.312
#>        excited.l1     0.084   0.184  -0.273   0.441
#>          upset.l1     0.317   0.123   0.077   0.558
#>         strong.l1    -0.063   0.171  -0.391   0.269
#>       stressed.l1     0.150   0.116  -0.076   0.379
#>          steps.l1     0.204   0.106  -0.013   0.409
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.110   0.182  -0.240   0.466
#>  disinterested.l1    -0.020   0.125  -0.271   0.225
#>        excited.l1     0.103   0.200  -0.292   0.501
#>          upset.l1    -0.092   0.132  -0.346   0.170
#>         strong.l1    -0.185   0.181  -0.534   0.173
#>       stressed.l1     0.130   0.123  -0.117   0.373
#>          steps.l1     0.038   0.116  -0.195   0.261
#> ---

# }
```
