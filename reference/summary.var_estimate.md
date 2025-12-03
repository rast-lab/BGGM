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
#>  interested--disinterested    -0.170   0.099  -0.346   0.039
#>        interested--excited     0.372   0.095   0.172   0.542
#>     disinterested--excited    -0.190   0.100  -0.386   0.011
#>          interested--upset    -0.205   0.099  -0.391  -0.013
#>       disinterested--upset    -0.036   0.100  -0.230   0.158
#>             excited--upset    -0.127   0.105  -0.326   0.078
#>         interested--strong     0.331   0.094   0.134   0.506
#>      disinterested--strong     0.100   0.101  -0.102   0.297
#>            excited--strong     0.495   0.080   0.331   0.637
#>              upset--strong     0.107   0.104  -0.104   0.297
#>       interested--stressed     0.271   0.096   0.078   0.459
#>    disinterested--stressed     0.142   0.094  -0.043   0.326
#>          excited--stressed    -0.157   0.107  -0.351   0.055
#>            upset--stressed     0.359   0.091   0.172   0.530
#>           strong--stressed    -0.015   0.106  -0.224   0.187
#>          interested--steps     0.074   0.103  -0.119   0.277
#>       disinterested--steps    -0.087   0.105  -0.312   0.111
#>             excited--steps    -0.014   0.110  -0.219   0.204
#>               upset--steps    -0.047   0.099  -0.237   0.147
#>              strong--steps     0.183   0.105  -0.041   0.382
#>            stressed--steps    -0.024   0.103  -0.226   0.178
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
#>     interested.l1     0.222   0.180  -0.133   0.562
#>  disinterested.l1    -0.051   0.121  -0.291   0.185
#>        excited.l1    -0.084   0.196  -0.467   0.306
#>          upset.l1    -0.156   0.127  -0.405   0.102
#>         strong.l1     0.031   0.175  -0.309   0.368
#>       stressed.l1    -0.020   0.120  -0.252   0.223
#>          steps.l1    -0.157   0.114  -0.375   0.068
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.016   0.175  -0.368   0.320
#>  disinterested.l1    -0.004   0.119  -0.241   0.226
#>        excited.l1    -0.178   0.193  -0.566   0.205
#>          upset.l1     0.259   0.125   0.016   0.503
#>         strong.l1     0.170   0.175  -0.175   0.509
#>       stressed.l1    -0.009   0.118  -0.240   0.222
#>          steps.l1     0.181   0.112  -0.037   0.405
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.178   0.184  -0.184   0.531
#>  disinterested.l1     0.054   0.123  -0.185   0.298
#>        excited.l1     0.004   0.201  -0.401   0.395
#>          upset.l1    -0.098   0.129  -0.355   0.155
#>         strong.l1     0.030   0.180  -0.323   0.390
#>       stressed.l1    -0.032   0.122  -0.271   0.207
#>          steps.l1    -0.210   0.116  -0.440   0.014
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.098   0.173  -0.441   0.239
#>  disinterested.l1    -0.016   0.115  -0.246   0.203
#>        excited.l1     0.053   0.186  -0.321   0.406
#>          upset.l1     0.431   0.121   0.198   0.666
#>         strong.l1     0.042   0.167  -0.288   0.356
#>       stressed.l1    -0.044   0.115  -0.269   0.178
#>          steps.l1     0.153   0.107  -0.066   0.362
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.173   0.184  -0.188   0.526
#>  disinterested.l1     0.049   0.125  -0.190   0.293
#>        excited.l1    -0.083   0.200  -0.479   0.315
#>          upset.l1     0.056   0.129  -0.197   0.308
#>         strong.l1     0.186   0.178  -0.169   0.529
#>       stressed.l1    -0.074   0.123  -0.319   0.164
#>          steps.l1    -0.093   0.116  -0.321   0.129
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.015   0.169  -0.314   0.351
#>  disinterested.l1     0.091   0.113  -0.131   0.315
#>        excited.l1     0.084   0.184  -0.273   0.436
#>          upset.l1     0.317   0.120   0.082   0.554
#>         strong.l1    -0.068   0.168  -0.397   0.260
#>       stressed.l1     0.152   0.114  -0.073   0.377
#>          steps.l1     0.204   0.107  -0.007   0.418
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.110   0.186  -0.253   0.478
#>  disinterested.l1    -0.024   0.127  -0.276   0.224
#>        excited.l1     0.099   0.197  -0.290   0.483
#>          upset.l1    -0.092   0.134  -0.358   0.168
#>         strong.l1    -0.180   0.182  -0.524   0.178
#>       stressed.l1     0.130   0.126  -0.116   0.378
#>          steps.l1     0.036   0.119  -0.198   0.274
#> ---

# }
```
