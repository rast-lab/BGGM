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
#>  interested--disinterested    -0.157   0.099  -0.348   0.035
#>        interested--excited     0.384   0.087   0.202   0.549
#>     disinterested--excited    -0.193   0.096  -0.368   0.007
#>          interested--upset    -0.207   0.103  -0.409  -0.011
#>       disinterested--upset    -0.030   0.099  -0.221   0.164
#>             excited--upset    -0.122   0.100  -0.319   0.072
#>         interested--strong     0.324   0.090   0.146   0.490
#>      disinterested--strong     0.098   0.103  -0.098   0.294
#>            excited--strong     0.491   0.077   0.331   0.630
#>              upset--strong     0.108   0.102  -0.096   0.307
#>       interested--stressed     0.276   0.099   0.086   0.464
#>    disinterested--stressed     0.146   0.101  -0.056   0.329
#>          excited--stressed    -0.168   0.099  -0.358   0.022
#>            upset--stressed     0.356   0.098   0.161   0.541
#>           strong--stressed    -0.014   0.111  -0.232   0.209
#>          interested--steps     0.080   0.104  -0.129   0.286
#>       disinterested--steps    -0.082   0.103  -0.281   0.129
#>             excited--steps    -0.001   0.099  -0.197   0.188
#>               upset--steps    -0.046   0.102  -0.241   0.161
#>              strong--steps     0.174   0.099  -0.024   0.354
#>            stressed--steps    -0.017   0.102  -0.210   0.182
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
#>     interested.l1     0.220   0.181  -0.134   0.582
#>  disinterested.l1    -0.049   0.123  -0.293   0.191
#>        excited.l1    -0.075   0.197  -0.461   0.311
#>          upset.l1    -0.150   0.128  -0.404   0.098
#>         strong.l1     0.023   0.174  -0.334   0.356
#>       stressed.l1    -0.022   0.122  -0.261   0.215
#>          steps.l1    -0.155   0.113  -0.378   0.068
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.013   0.178  -0.362   0.332
#>  disinterested.l1    -0.007   0.122  -0.251   0.230
#>        excited.l1    -0.186   0.193  -0.568   0.195
#>          upset.l1     0.259   0.128   0.010   0.510
#>         strong.l1     0.176   0.172  -0.165   0.510
#>       stressed.l1    -0.011   0.120  -0.249   0.228
#>          steps.l1     0.181   0.111  -0.036   0.397
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.178   0.183  -0.181   0.538
#>  disinterested.l1     0.056   0.124  -0.190   0.294
#>        excited.l1     0.010   0.195  -0.374   0.389
#>          upset.l1    -0.092   0.131  -0.351   0.164
#>         strong.l1     0.022   0.177  -0.332   0.371
#>       stressed.l1    -0.033   0.123  -0.273   0.205
#>          steps.l1    -0.207   0.114  -0.433   0.016
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.096   0.173  -0.441   0.248
#>  disinterested.l1    -0.020   0.115  -0.244   0.210
#>        excited.l1     0.045   0.187  -0.328   0.413
#>          upset.l1     0.430   0.124   0.186   0.675
#>         strong.l1     0.052   0.172  -0.286   0.391
#>       stressed.l1    -0.048   0.116  -0.280   0.178
#>          steps.l1     0.151   0.106  -0.061   0.358
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.173   0.185  -0.187   0.531
#>  disinterested.l1     0.050   0.125  -0.192   0.301
#>        excited.l1    -0.075   0.198  -0.458   0.315
#>          upset.l1     0.061   0.132  -0.202   0.323
#>         strong.l1     0.178   0.179  -0.181   0.526
#>       stressed.l1    -0.077   0.123  -0.321   0.170
#>          steps.l1    -0.091   0.116  -0.316   0.136
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.014   0.174  -0.323   0.356
#>  disinterested.l1     0.088   0.118  -0.141   0.314
#>        excited.l1     0.085   0.190  -0.282   0.467
#>          upset.l1     0.318   0.125   0.078   0.570
#>         strong.l1    -0.063   0.172  -0.401   0.276
#>       stressed.l1     0.151   0.118  -0.078   0.379
#>          steps.l1     0.202   0.108  -0.009   0.415
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.107   0.185  -0.257   0.472
#>  disinterested.l1    -0.024   0.126  -0.269   0.223
#>        excited.l1     0.106   0.202  -0.283   0.501
#>          upset.l1    -0.090   0.133  -0.352   0.169
#>         strong.l1    -0.185   0.180  -0.533   0.181
#>       stressed.l1     0.132   0.125  -0.110   0.374
#>          steps.l1     0.039   0.117  -0.192   0.268
#> ---

# }
```
