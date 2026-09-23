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
#>  interested--disinterested    -0.180   0.104  -0.378   0.022
#>        interested--excited     0.373   0.091   0.182   0.536
#>     disinterested--excited    -0.171   0.106  -0.371   0.045
#>          interested--upset    -0.224   0.098  -0.414  -0.034
#>       disinterested--upset    -0.038   0.108  -0.246   0.166
#>             excited--upset    -0.125   0.107  -0.339   0.082
#>         interested--strong     0.330   0.092   0.141   0.511
#>      disinterested--strong     0.090   0.111  -0.136   0.296
#>            excited--strong     0.493   0.082   0.314   0.638
#>              upset--strong     0.120   0.110  -0.089   0.333
#>       interested--stressed     0.288   0.099   0.082   0.471
#>    disinterested--stressed     0.155   0.104  -0.058   0.355
#>          excited--stressed    -0.173   0.110  -0.389   0.048
#>            upset--stressed     0.357   0.086   0.185   0.521
#>           strong--stressed    -0.016   0.106  -0.215   0.200
#>          interested--steps     0.074   0.102  -0.140   0.268
#>       disinterested--steps    -0.091   0.103  -0.290   0.098
#>             excited--steps    -0.013   0.102  -0.209   0.192
#>               upset--steps    -0.045   0.106  -0.250   0.162
#>              strong--steps     0.181   0.101  -0.018   0.379
#>            stressed--steps    -0.017   0.100  -0.206   0.183
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
#>     interested.l1     0.223   0.179  -0.134   0.572
#>  disinterested.l1    -0.046   0.122  -0.285   0.189
#>        excited.l1    -0.083   0.199  -0.460   0.313
#>          upset.l1    -0.153   0.127  -0.395   0.099
#>         strong.l1     0.029   0.179  -0.316   0.382
#>       stressed.l1    -0.020   0.119  -0.253   0.216
#>          steps.l1    -0.156   0.113  -0.382   0.070
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.013   0.176  -0.356   0.328
#>  disinterested.l1    -0.004   0.121  -0.243   0.231
#>        excited.l1    -0.182   0.192  -0.558   0.195
#>          upset.l1     0.259   0.128   0.008   0.511
#>         strong.l1     0.172   0.173  -0.161   0.515
#>       stressed.l1    -0.012   0.120  -0.248   0.227
#>          steps.l1     0.180   0.114  -0.047   0.404
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.180   0.184  -0.173   0.536
#>  disinterested.l1     0.059   0.122  -0.182   0.300
#>        excited.l1     0.002   0.197  -0.384   0.382
#>          upset.l1    -0.095   0.130  -0.348   0.164
#>         strong.l1     0.029   0.181  -0.337   0.389
#>       stressed.l1    -0.032   0.121  -0.270   0.208
#>          steps.l1    -0.207   0.114  -0.426   0.024
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.100   0.172  -0.435   0.246
#>  disinterested.l1    -0.022   0.117  -0.251   0.205
#>        excited.l1     0.053   0.188  -0.319   0.421
#>          upset.l1     0.428   0.123   0.186   0.676
#>         strong.l1     0.049   0.170  -0.285   0.375
#>       stressed.l1    -0.043   0.116  -0.271   0.178
#>          steps.l1     0.146   0.107  -0.061   0.358
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.177   0.183  -0.188   0.532
#>  disinterested.l1     0.052   0.124  -0.191   0.289
#>        excited.l1    -0.083   0.201  -0.472   0.320
#>          upset.l1     0.056   0.129  -0.193   0.313
#>         strong.l1     0.183   0.183  -0.175   0.541
#>       stressed.l1    -0.074   0.121  -0.313   0.162
#>          steps.l1    -0.093   0.116  -0.316   0.133
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.017   0.172  -0.320   0.351
#>  disinterested.l1     0.089   0.117  -0.138   0.316
#>        excited.l1     0.078   0.188  -0.286   0.452
#>          upset.l1     0.316   0.122   0.079   0.552
#>         strong.l1    -0.061   0.168  -0.385   0.267
#>       stressed.l1     0.151   0.114  -0.075   0.371
#>          steps.l1     0.201   0.109  -0.009   0.418
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.111   0.183  -0.245   0.461
#>  disinterested.l1    -0.020   0.123  -0.260   0.227
#>        excited.l1     0.099   0.202  -0.300   0.497
#>          upset.l1    -0.094   0.131  -0.346   0.168
#>         strong.l1    -0.183   0.181  -0.540   0.168
#>       stressed.l1     0.130   0.123  -0.115   0.369
#>          steps.l1     0.038   0.116  -0.186   0.260
#> ---

# }
```
