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
#>  interested--disinterested    -0.167   0.101  -0.365   0.034
#>        interested--excited     0.393   0.093   0.205   0.564
#>     disinterested--excited    -0.172   0.098  -0.358   0.020
#>          interested--upset    -0.202   0.100  -0.395   0.006
#>       disinterested--upset    -0.024   0.099  -0.220   0.165
#>             excited--upset    -0.126   0.104  -0.337   0.079
#>         interested--strong     0.326   0.094   0.130   0.502
#>      disinterested--strong     0.098   0.101  -0.104   0.290
#>            excited--strong     0.484   0.091   0.284   0.639
#>              upset--strong     0.116   0.103  -0.094   0.309
#>       interested--stressed     0.273   0.091   0.086   0.446
#>    disinterested--stressed     0.143   0.099  -0.047   0.340
#>          excited--stressed    -0.163   0.095  -0.337   0.027
#>            upset--stressed     0.355   0.094   0.156   0.529
#>           strong--stressed    -0.022   0.104  -0.221   0.189
#>          interested--steps     0.072   0.109  -0.146   0.282
#>       disinterested--steps    -0.086   0.109  -0.292   0.127
#>             excited--steps     0.000   0.110  -0.214   0.212
#>               upset--steps    -0.056   0.109  -0.261   0.162
#>              strong--steps     0.180   0.103  -0.028   0.368
#>            stressed--steps    -0.017   0.105  -0.219   0.197
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
#>     interested.l1     0.224   0.180  -0.125   0.575
#>  disinterested.l1    -0.050   0.124  -0.292   0.192
#>        excited.l1    -0.078   0.199  -0.472   0.298
#>          upset.l1    -0.149   0.130  -0.406   0.097
#>         strong.l1     0.023   0.177  -0.315   0.370
#>       stressed.l1    -0.022   0.121  -0.263   0.221
#>          steps.l1    -0.153   0.113  -0.375   0.065
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.018   0.177  -0.376   0.328
#>  disinterested.l1    -0.005   0.122  -0.243   0.242
#>        excited.l1    -0.181   0.194  -0.559   0.194
#>          upset.l1     0.260   0.128   0.003   0.514
#>         strong.l1     0.176   0.174  -0.168   0.516
#>       stressed.l1    -0.010   0.122  -0.249   0.230
#>          steps.l1     0.181   0.114  -0.043   0.405
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.182   0.184  -0.174   0.546
#>  disinterested.l1     0.056   0.126  -0.200   0.297
#>        excited.l1     0.007   0.203  -0.393   0.393
#>          upset.l1    -0.094   0.131  -0.353   0.162
#>         strong.l1     0.020   0.186  -0.341   0.382
#>       stressed.l1    -0.033   0.123  -0.277   0.207
#>          steps.l1    -0.206   0.115  -0.436   0.019
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.100   0.171  -0.437   0.238
#>  disinterested.l1    -0.018   0.115  -0.244   0.206
#>        excited.l1     0.050   0.185  -0.316   0.408
#>          upset.l1     0.429   0.123   0.186   0.665
#>         strong.l1     0.049   0.167  -0.274   0.370
#>       stressed.l1    -0.044   0.115  -0.265   0.182
#>          steps.l1     0.151   0.109  -0.061   0.362
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.175   0.187  -0.195   0.550
#>  disinterested.l1     0.048   0.127  -0.201   0.293
#>        excited.l1    -0.081   0.204  -0.483   0.311
#>          upset.l1     0.058   0.134  -0.205   0.319
#>         strong.l1     0.181   0.187  -0.176   0.552
#>       stressed.l1    -0.076   0.126  -0.321   0.169
#>          steps.l1    -0.090   0.115  -0.318   0.139
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.010   0.172  -0.337   0.350
#>  disinterested.l1     0.089   0.117  -0.137   0.320
#>        excited.l1     0.090   0.187  -0.273   0.464
#>          upset.l1     0.318   0.124   0.075   0.562
#>         strong.l1    -0.067   0.168  -0.403   0.259
#>       stressed.l1     0.153   0.116  -0.076   0.375
#>          steps.l1     0.204   0.109  -0.015   0.415
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.109   0.187  -0.253   0.472
#>  disinterested.l1    -0.021   0.127  -0.273   0.232
#>        excited.l1     0.102   0.202  -0.294   0.506
#>          upset.l1    -0.091   0.134  -0.350   0.174
#>         strong.l1    -0.184   0.182  -0.551   0.173
#>       stressed.l1     0.129   0.127  -0.119   0.373
#>          steps.l1     0.041   0.116  -0.192   0.268
#> ---

# }
```
