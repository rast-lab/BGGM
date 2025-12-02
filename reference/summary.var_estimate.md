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
#>  interested--disinterested    -0.182   0.109  -0.393   0.023
#>        interested--excited     0.386   0.096   0.188   0.565
#>     disinterested--excited    -0.178   0.104  -0.378   0.022
#>          interested--upset    -0.199   0.102  -0.393   0.011
#>       disinterested--upset    -0.037   0.103  -0.249   0.165
#>             excited--upset    -0.133   0.100  -0.328   0.061
#>         interested--strong     0.324   0.097   0.139   0.510
#>      disinterested--strong     0.112   0.108  -0.101   0.319
#>            excited--strong     0.494   0.084   0.316   0.643
#>              upset--strong     0.109   0.108  -0.114   0.308
#>       interested--stressed     0.277   0.098   0.072   0.449
#>    disinterested--stressed     0.153   0.102  -0.052   0.338
#>          excited--stressed    -0.170   0.100  -0.368   0.025
#>            upset--stressed     0.355   0.093   0.164   0.527
#>           strong--stressed    -0.011   0.108  -0.225   0.205
#>          interested--steps     0.079   0.103  -0.120   0.281
#>       disinterested--steps    -0.088   0.103  -0.292   0.120
#>             excited--steps    -0.016   0.107  -0.221   0.184
#>               upset--steps    -0.053   0.109  -0.256   0.174
#>              strong--steps     0.173   0.098  -0.027   0.355
#>            stressed--steps    -0.016   0.105  -0.227   0.191
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
#>     interested.l1     0.223   0.180  -0.129   0.580
#>  disinterested.l1    -0.048   0.122  -0.288   0.188
#>        excited.l1    -0.083   0.196  -0.465   0.312
#>          upset.l1    -0.152   0.129  -0.409   0.098
#>         strong.l1     0.030   0.173  -0.308   0.369
#>       stressed.l1    -0.021   0.120  -0.253   0.217
#>          steps.l1    -0.157   0.113  -0.382   0.060
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.017   0.178  -0.361   0.330
#>  disinterested.l1    -0.005   0.122  -0.245   0.233
#>        excited.l1    -0.182   0.196  -0.566   0.216
#>          upset.l1     0.258   0.128   0.002   0.506
#>         strong.l1     0.173   0.174  -0.173   0.510
#>       stressed.l1    -0.010   0.121  -0.251   0.225
#>          steps.l1     0.180   0.112  -0.040   0.404
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.179   0.181  -0.174   0.534
#>  disinterested.l1     0.057   0.123  -0.188   0.299
#>        excited.l1     0.002   0.199  -0.387   0.398
#>          upset.l1    -0.095   0.130  -0.353   0.155
#>         strong.l1     0.031   0.176  -0.319   0.376
#>       stressed.l1    -0.031   0.123  -0.271   0.211
#>          steps.l1    -0.210   0.113  -0.437   0.013
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.098   0.174  -0.437   0.237
#>  disinterested.l1    -0.018   0.116  -0.249   0.205
#>        excited.l1     0.052   0.187  -0.313   0.421
#>          upset.l1     0.427   0.124   0.175   0.663
#>         strong.l1     0.048   0.169  -0.277   0.373
#>       stressed.l1    -0.043   0.114  -0.269   0.186
#>          steps.l1     0.151   0.108  -0.060   0.360
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.177   0.183  -0.173   0.541
#>  disinterested.l1     0.050   0.125  -0.202   0.285
#>        excited.l1    -0.086   0.198  -0.475   0.306
#>          upset.l1     0.058   0.130  -0.192   0.308
#>         strong.l1     0.184   0.177  -0.167   0.526
#>       stressed.l1    -0.074   0.122  -0.312   0.162
#>          steps.l1    -0.093   0.114  -0.324   0.127
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.014   0.171  -0.322   0.351
#>  disinterested.l1     0.091   0.116  -0.136   0.317
#>        excited.l1     0.091   0.184  -0.269   0.443
#>          upset.l1     0.315   0.122   0.079   0.552
#>         strong.l1    -0.071   0.169  -0.408   0.261
#>       stressed.l1     0.155   0.116  -0.076   0.377
#>          steps.l1     0.205   0.109  -0.014   0.420
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.109   0.183  -0.256   0.461
#>  disinterested.l1    -0.023   0.126  -0.276   0.224
#>        excited.l1     0.099   0.203  -0.301   0.497
#>          upset.l1    -0.091   0.132  -0.351   0.170
#>         strong.l1    -0.183   0.182  -0.536   0.179
#>       stressed.l1     0.129   0.124  -0.119   0.369
#>          steps.l1     0.041   0.118  -0.189   0.272
#> ---

# }
```
