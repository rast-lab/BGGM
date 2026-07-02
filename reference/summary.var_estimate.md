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
#>  interested--disinterested    -0.170   0.105  -0.361   0.062
#>        interested--excited     0.381   0.092   0.190   0.548
#>     disinterested--excited    -0.183   0.098  -0.369   0.017
#>          interested--upset    -0.211   0.096  -0.400  -0.025
#>       disinterested--upset    -0.040   0.104  -0.246   0.158
#>             excited--upset    -0.135   0.102  -0.341   0.059
#>         interested--strong     0.324   0.090   0.142   0.484
#>      disinterested--strong     0.103   0.107  -0.106   0.323
#>            excited--strong     0.492   0.081   0.326   0.638
#>              upset--strong     0.118   0.102  -0.082   0.318
#>       interested--stressed     0.283   0.091   0.102   0.455
#>    disinterested--stressed     0.155   0.105  -0.063   0.357
#>          excited--stressed    -0.169   0.103  -0.363   0.028
#>            upset--stressed     0.352   0.089   0.177   0.521
#>           strong--stressed    -0.013   0.102  -0.222   0.184
#>          interested--steps     0.067   0.105  -0.155   0.255
#>       disinterested--steps    -0.082   0.108  -0.293   0.130
#>             excited--steps    -0.014   0.101  -0.204   0.195
#>               upset--steps    -0.050   0.102  -0.249   0.152
#>              strong--steps     0.190   0.098  -0.012   0.375
#>            stressed--steps    -0.032   0.100  -0.234   0.172
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
#>     interested.l1     0.223   0.176  -0.117   0.566
#>  disinterested.l1    -0.051   0.120  -0.284   0.190
#>        excited.l1    -0.085   0.193  -0.465   0.293
#>          upset.l1    -0.153   0.125  -0.398   0.089
#>         strong.l1     0.025   0.174  -0.314   0.364
#>       stressed.l1    -0.019   0.122  -0.258   0.219
#>          steps.l1    -0.152   0.113  -0.370   0.074
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.011   0.180  -0.374   0.332
#>  disinterested.l1    -0.004   0.121  -0.249   0.229
#>        excited.l1    -0.186   0.195  -0.565   0.206
#>          upset.l1     0.258   0.128   0.012   0.509
#>         strong.l1     0.176   0.176  -0.168   0.522
#>       stressed.l1    -0.010   0.122  -0.246   0.230
#>          steps.l1     0.179   0.114  -0.044   0.407
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.181   0.177  -0.171   0.528
#>  disinterested.l1     0.055   0.123  -0.186   0.294
#>        excited.l1     0.003   0.199  -0.390   0.387
#>          upset.l1    -0.096   0.128  -0.343   0.156
#>         strong.l1     0.023   0.175  -0.324   0.365
#>       stressed.l1    -0.032   0.124  -0.273   0.209
#>          steps.l1    -0.205   0.114  -0.425   0.019
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.097   0.170  -0.429   0.237
#>  disinterested.l1    -0.018   0.115  -0.241   0.208
#>        excited.l1     0.052   0.189  -0.318   0.420
#>          upset.l1     0.428   0.125   0.184   0.666
#>         strong.l1     0.046   0.167  -0.287   0.373
#>       stressed.l1    -0.045   0.114  -0.270   0.180
#>          steps.l1     0.151   0.107  -0.057   0.360
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.176   0.180  -0.172   0.527
#>  disinterested.l1     0.046   0.123  -0.191   0.289
#>        excited.l1    -0.084   0.199  -0.484   0.294
#>          upset.l1     0.055   0.129  -0.195   0.308
#>         strong.l1     0.180   0.179  -0.173   0.533
#>       stressed.l1    -0.072   0.124  -0.315   0.172
#>          steps.l1    -0.091   0.116  -0.308   0.144
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.011   0.169  -0.315   0.348
#>  disinterested.l1     0.088   0.117  -0.147   0.314
#>        excited.l1     0.083   0.190  -0.291   0.454
#>          upset.l1     0.315   0.124   0.069   0.560
#>         strong.l1    -0.062   0.166  -0.389   0.278
#>       stressed.l1     0.154   0.116  -0.072   0.383
#>          steps.l1     0.207   0.108  -0.005   0.418
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.111   0.181  -0.246   0.462
#>  disinterested.l1    -0.023   0.123  -0.266   0.222
#>        excited.l1     0.100   0.199  -0.288   0.489
#>          upset.l1    -0.093   0.131  -0.347   0.161
#>         strong.l1    -0.186   0.180  -0.541   0.172
#>       stressed.l1     0.132   0.125  -0.108   0.375
#>          steps.l1     0.041   0.115  -0.190   0.266
#> ---

# }
```
