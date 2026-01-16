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
#>  interested--disinterested    -0.180   0.100  -0.369   0.015
#>        interested--excited     0.392   0.088   0.208   0.550
#>     disinterested--excited    -0.173   0.101  -0.365   0.029
#>          interested--upset    -0.198   0.106  -0.391   0.018
#>       disinterested--upset    -0.044   0.113  -0.250   0.190
#>             excited--upset    -0.136   0.105  -0.347   0.062
#>         interested--strong     0.324   0.088   0.146   0.483
#>      disinterested--strong     0.109   0.107  -0.096   0.314
#>            excited--strong     0.488   0.079   0.322   0.627
#>              upset--strong     0.114   0.108  -0.095   0.323
#>       interested--stressed     0.278   0.096   0.075   0.451
#>    disinterested--stressed     0.162   0.100  -0.038   0.353
#>          excited--stressed    -0.163   0.102  -0.353   0.041
#>            upset--stressed     0.357   0.097   0.147   0.528
#>           strong--stressed    -0.023   0.103  -0.220   0.194
#>          interested--steps     0.070   0.100  -0.120   0.262
#>       disinterested--steps    -0.094   0.100  -0.289   0.104
#>             excited--steps    -0.018   0.105  -0.228   0.187
#>               upset--steps    -0.055   0.105  -0.269   0.144
#>              strong--steps     0.199   0.100  -0.008   0.373
#>            stressed--steps    -0.003   0.107  -0.211   0.215
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
#>     interested.l1     0.219   0.178  -0.134   0.562
#>  disinterested.l1    -0.051   0.122  -0.293   0.195
#>        excited.l1    -0.079   0.199  -0.476   0.311
#>          upset.l1    -0.152   0.129  -0.406   0.098
#>         strong.l1     0.024   0.176  -0.329   0.360
#>       stressed.l1    -0.017   0.123  -0.259   0.226
#>          steps.l1    -0.154   0.113  -0.378   0.068
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.014   0.176  -0.359   0.325
#>  disinterested.l1    -0.001   0.120  -0.241   0.229
#>        excited.l1    -0.184   0.197  -0.568   0.207
#>          upset.l1     0.254   0.130  -0.001   0.507
#>         strong.l1     0.175   0.174  -0.173   0.520
#>       stressed.l1    -0.012   0.121  -0.252   0.224
#>          steps.l1     0.181   0.111  -0.041   0.396
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.177   0.179  -0.176   0.525
#>  disinterested.l1     0.055   0.123  -0.188   0.297
#>        excited.l1     0.007   0.202  -0.390   0.401
#>          upset.l1    -0.095   0.131  -0.353   0.160
#>         strong.l1     0.025   0.180  -0.327   0.385
#>       stressed.l1    -0.028   0.125  -0.275   0.217
#>          steps.l1    -0.207   0.116  -0.438   0.015
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.097   0.170  -0.436   0.233
#>  disinterested.l1    -0.017   0.117  -0.249   0.213
#>        excited.l1     0.052   0.188  -0.318   0.421
#>          upset.l1     0.430   0.126   0.186   0.677
#>         strong.l1     0.047   0.169  -0.280   0.389
#>       stressed.l1    -0.048   0.118  -0.278   0.180
#>          steps.l1     0.152   0.109  -0.061   0.367
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.173   0.181  -0.190   0.523
#>  disinterested.l1     0.048   0.124  -0.195   0.294
#>        excited.l1    -0.078   0.202  -0.477   0.305
#>          upset.l1     0.057   0.130  -0.199   0.311
#>         strong.l1     0.178   0.181  -0.186   0.536
#>       stressed.l1    -0.072   0.123  -0.314   0.171
#>          steps.l1    -0.090   0.117  -0.322   0.132
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.012   0.169  -0.325   0.339
#>  disinterested.l1     0.089   0.118  -0.150   0.319
#>        excited.l1     0.087   0.186  -0.282   0.447
#>          upset.l1     0.314   0.125   0.064   0.559
#>         strong.l1    -0.065   0.167  -0.390   0.273
#>       stressed.l1     0.152   0.116  -0.073   0.382
#>          steps.l1     0.204   0.109  -0.008   0.420
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.109   0.187  -0.252   0.475
#>  disinterested.l1    -0.025   0.126  -0.277   0.218
#>        excited.l1     0.099   0.203  -0.310   0.491
#>          upset.l1    -0.089   0.133  -0.353   0.174
#>         strong.l1    -0.183   0.182  -0.545   0.176
#>       stressed.l1     0.129   0.126  -0.112   0.383
#>          steps.l1     0.039   0.117  -0.188   0.269
#> ---

# }
```
