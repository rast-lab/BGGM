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
#>  interested--disinterested    -0.168   0.096  -0.355   0.017
#>        interested--excited     0.390   0.094   0.196   0.559
#>     disinterested--excited    -0.187   0.097  -0.375   0.007
#>          interested--upset    -0.206   0.096  -0.386  -0.026
#>       disinterested--upset    -0.031   0.097  -0.216   0.167
#>             excited--upset    -0.130   0.107  -0.336   0.092
#>         interested--strong     0.328   0.093   0.141   0.504
#>      disinterested--strong     0.108   0.106  -0.108   0.310
#>            excited--strong     0.486   0.083   0.314   0.632
#>              upset--strong     0.108   0.106  -0.094   0.327
#>       interested--stressed     0.283   0.095   0.090   0.460
#>    disinterested--stressed     0.149   0.096  -0.042   0.330
#>          excited--stressed    -0.174   0.098  -0.369   0.015
#>            upset--stressed     0.348   0.095   0.152   0.516
#>           strong--stressed    -0.011   0.110  -0.225   0.204
#>          interested--steps     0.079   0.106  -0.120   0.290
#>       disinterested--steps    -0.086   0.110  -0.287   0.129
#>             excited--steps    -0.018   0.104  -0.222   0.185
#>               upset--steps    -0.037   0.106  -0.241   0.174
#>              strong--steps     0.188   0.099  -0.012   0.375
#>            stressed--steps    -0.027   0.111  -0.243   0.185
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
#>     interested.l1     0.223   0.181  -0.133   0.577
#>  disinterested.l1    -0.052   0.122  -0.289   0.195
#>        excited.l1    -0.081   0.193  -0.456   0.297
#>          upset.l1    -0.151   0.130  -0.410   0.105
#>         strong.l1     0.021   0.178  -0.335   0.362
#>       stressed.l1    -0.020   0.122  -0.255   0.224
#>          steps.l1    -0.154   0.114  -0.373   0.068
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.013   0.178  -0.358   0.342
#>  disinterested.l1    -0.003   0.123  -0.243   0.231
#>        excited.l1    -0.184   0.197  -0.572   0.195
#>          upset.l1     0.258   0.128   0.008   0.510
#>         strong.l1     0.176   0.172  -0.159   0.520
#>       stressed.l1    -0.011   0.122  -0.250   0.226
#>          steps.l1     0.182   0.115  -0.044   0.403
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.179   0.183  -0.169   0.547
#>  disinterested.l1     0.052   0.125  -0.190   0.292
#>        excited.l1     0.004   0.199  -0.382   0.390
#>          upset.l1    -0.095   0.128  -0.348   0.158
#>         strong.l1     0.022   0.182  -0.338   0.377
#>       stressed.l1    -0.030   0.124  -0.273   0.221
#>          steps.l1    -0.208   0.117  -0.437   0.018
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.097   0.169  -0.429   0.236
#>  disinterested.l1    -0.018   0.118  -0.254   0.211
#>        excited.l1     0.051   0.188  -0.326   0.420
#>          upset.l1     0.430   0.122   0.191   0.672
#>         strong.l1     0.047   0.167  -0.279   0.379
#>       stressed.l1    -0.045   0.117  -0.280   0.178
#>          steps.l1     0.152   0.108  -0.062   0.365
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.176   0.183  -0.183   0.541
#>  disinterested.l1     0.047   0.127  -0.206   0.290
#>        excited.l1    -0.083   0.199  -0.467   0.309
#>          upset.l1     0.058   0.131  -0.204   0.316
#>         strong.l1     0.179   0.180  -0.177   0.534
#>       stressed.l1    -0.073   0.125  -0.317   0.176
#>          steps.l1    -0.092   0.116  -0.319   0.131
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.014   0.171  -0.315   0.343
#>  disinterested.l1     0.090   0.118  -0.140   0.324
#>        excited.l1     0.086   0.191  -0.289   0.458
#>          upset.l1     0.316   0.124   0.073   0.560
#>         strong.l1    -0.067   0.169  -0.393   0.263
#>       stressed.l1     0.152   0.116  -0.076   0.373
#>          steps.l1     0.205   0.108  -0.006   0.417
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.111   0.185  -0.243   0.476
#>  disinterested.l1    -0.022   0.125  -0.272   0.227
#>        excited.l1     0.101   0.199  -0.294   0.476
#>          upset.l1    -0.092   0.132  -0.356   0.171
#>         strong.l1    -0.188   0.180  -0.537   0.156
#>       stressed.l1     0.130   0.124  -0.115   0.373
#>          steps.l1     0.041   0.116  -0.184   0.270
#> ---

# }
```
