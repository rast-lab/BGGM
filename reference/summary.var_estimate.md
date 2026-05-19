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
#>  interested--disinterested    -0.173   0.104  -0.365   0.041
#>        interested--excited     0.383   0.088   0.203   0.546
#>     disinterested--excited    -0.178   0.105  -0.372   0.049
#>          interested--upset    -0.211   0.097  -0.393  -0.019
#>       disinterested--upset    -0.033   0.105  -0.232   0.172
#>             excited--upset    -0.131   0.104  -0.332   0.079
#>         interested--strong     0.331   0.092   0.132   0.502
#>      disinterested--strong     0.097   0.098  -0.101   0.280
#>            excited--strong     0.487   0.080   0.319   0.630
#>              upset--strong     0.112   0.102  -0.093   0.295
#>       interested--stressed     0.274   0.095   0.088   0.455
#>    disinterested--stressed     0.151   0.101  -0.052   0.350
#>          excited--stressed    -0.170   0.092  -0.342   0.021
#>            upset--stressed     0.351   0.090   0.171   0.523
#>           strong--stressed    -0.010   0.102  -0.206   0.194
#>          interested--steps     0.076   0.102  -0.131   0.274
#>       disinterested--steps    -0.078   0.098  -0.271   0.111
#>             excited--steps    -0.006   0.108  -0.226   0.206
#>               upset--steps    -0.035   0.106  -0.238   0.171
#>              strong--steps     0.177   0.098  -0.030   0.358
#>            stressed--steps    -0.015   0.106  -0.226   0.184
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
#>     interested.l1     0.221   0.180  -0.136   0.576
#>  disinterested.l1    -0.048   0.122  -0.291   0.185
#>        excited.l1    -0.084   0.196  -0.471   0.306
#>          upset.l1    -0.153   0.132  -0.414   0.106
#>         strong.l1     0.027   0.176  -0.308   0.379
#>       stressed.l1    -0.022   0.122  -0.263   0.215
#>          steps.l1    -0.153   0.113  -0.372   0.069
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.009   0.181  -0.360   0.347
#>  disinterested.l1    -0.004   0.121  -0.242   0.237
#>        excited.l1    -0.185   0.197  -0.577   0.201
#>          upset.l1     0.261   0.129   0.008   0.521
#>         strong.l1     0.173   0.176  -0.170   0.515
#>       stressed.l1    -0.010   0.122  -0.248   0.228
#>          steps.l1     0.179   0.112  -0.045   0.395
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.178   0.181  -0.181   0.541
#>  disinterested.l1     0.057   0.124  -0.190   0.302
#>        excited.l1     0.005   0.196  -0.390   0.389
#>          upset.l1    -0.097   0.130  -0.353   0.161
#>         strong.l1     0.025   0.177  -0.314   0.372
#>       stressed.l1    -0.033   0.123  -0.270   0.208
#>          steps.l1    -0.206   0.114  -0.428   0.021
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.101   0.171  -0.439   0.247
#>  disinterested.l1    -0.020   0.117  -0.257   0.212
#>        excited.l1     0.052   0.190  -0.325   0.428
#>          upset.l1     0.427   0.124   0.185   0.665
#>         strong.l1     0.049   0.167  -0.275   0.383
#>       stressed.l1    -0.041   0.116  -0.265   0.185
#>          steps.l1     0.153   0.108  -0.061   0.356
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.174   0.182  -0.183   0.535
#>  disinterested.l1     0.050   0.123  -0.197   0.283
#>        excited.l1    -0.083   0.200  -0.467   0.312
#>          upset.l1     0.054   0.132  -0.204   0.311
#>         strong.l1     0.181   0.181  -0.176   0.544
#>       stressed.l1    -0.074   0.122  -0.313   0.165
#>          steps.l1    -0.089   0.115  -0.310   0.139
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.014   0.168  -0.313   0.346
#>  disinterested.l1     0.091   0.114  -0.131   0.310
#>        excited.l1     0.084   0.184  -0.271   0.442
#>          upset.l1     0.314   0.123   0.069   0.552
#>         strong.l1    -0.064   0.165  -0.385   0.259
#>       stressed.l1     0.155   0.114  -0.069   0.385
#>          steps.l1     0.205   0.107  -0.004   0.417
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.107   0.188  -0.266   0.484
#>  disinterested.l1    -0.022   0.126  -0.272   0.222
#>        excited.l1     0.099   0.202  -0.303   0.502
#>          upset.l1    -0.096   0.134  -0.359   0.171
#>         strong.l1    -0.183   0.182  -0.541   0.178
#>       stressed.l1     0.133   0.125  -0.111   0.384
#>          steps.l1     0.043   0.114  -0.180   0.267
#> ---

# }
```
