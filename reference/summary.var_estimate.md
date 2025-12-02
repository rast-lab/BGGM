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
#>  interested--disinterested    -0.168   0.101  -0.356   0.031
#>        interested--excited     0.390   0.087   0.218   0.554
#>     disinterested--excited    -0.177   0.106  -0.383   0.034
#>          interested--upset    -0.199   0.101  -0.382   0.008
#>       disinterested--upset    -0.033   0.103  -0.237   0.162
#>             excited--upset    -0.134   0.101  -0.335   0.058
#>         interested--strong     0.316   0.095   0.125   0.497
#>      disinterested--strong     0.105   0.104  -0.106   0.310
#>            excited--strong     0.496   0.077   0.339   0.641
#>              upset--strong     0.113   0.100  -0.085   0.306
#>       interested--stressed     0.270   0.103   0.057   0.456
#>    disinterested--stressed     0.150   0.100  -0.053   0.337
#>          excited--stressed    -0.168   0.096  -0.352   0.024
#>            upset--stressed     0.352   0.091   0.166   0.523
#>           strong--stressed    -0.003   0.101  -0.199   0.198
#>          interested--steps     0.082   0.106  -0.138   0.278
#>       disinterested--steps    -0.087   0.108  -0.289   0.137
#>             excited--steps    -0.016   0.108  -0.224   0.195
#>               upset--steps    -0.045   0.104  -0.244   0.168
#>              strong--steps     0.183   0.107  -0.031   0.380
#>            stressed--steps    -0.027   0.101  -0.217   0.178
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
#>     interested.l1     0.220   0.177  -0.129   0.570
#>  disinterested.l1    -0.052   0.122  -0.296   0.187
#>        excited.l1    -0.079   0.190  -0.459   0.300
#>          upset.l1    -0.152   0.128  -0.402   0.100
#>         strong.l1     0.026   0.174  -0.306   0.382
#>       stressed.l1    -0.019   0.125  -0.261   0.228
#>          steps.l1    -0.154   0.114  -0.379   0.066
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.013   0.178  -0.364   0.337
#>  disinterested.l1    -0.004   0.122  -0.243   0.238
#>        excited.l1    -0.181   0.195  -0.558   0.205
#>          upset.l1     0.258   0.130   0.003   0.509
#>         strong.l1     0.174   0.174  -0.162   0.510
#>       stressed.l1    -0.009   0.122  -0.240   0.235
#>          steps.l1     0.181   0.115  -0.045   0.413
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.177   0.177  -0.167   0.533
#>  disinterested.l1     0.052   0.122  -0.193   0.290
#>        excited.l1     0.004   0.191  -0.375   0.376
#>          upset.l1    -0.096   0.129  -0.348   0.163
#>         strong.l1     0.026   0.173  -0.314   0.359
#>       stressed.l1    -0.031   0.124  -0.279   0.210
#>          steps.l1    -0.208   0.114  -0.432   0.015
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.098   0.172  -0.434   0.242
#>  disinterested.l1    -0.017   0.118  -0.251   0.209
#>        excited.l1     0.049   0.189  -0.318   0.419
#>          upset.l1     0.427   0.123   0.192   0.669
#>         strong.l1     0.051   0.168  -0.276   0.379
#>       stressed.l1    -0.043   0.119  -0.278   0.190
#>          steps.l1     0.151   0.108  -0.058   0.368
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.172   0.181  -0.177   0.530
#>  disinterested.l1     0.046   0.124  -0.205   0.281
#>        excited.l1    -0.081   0.194  -0.461   0.302
#>          upset.l1     0.058   0.131  -0.195   0.311
#>         strong.l1     0.182   0.177  -0.159   0.535
#>       stressed.l1    -0.072   0.126  -0.326   0.178
#>          steps.l1    -0.092   0.116  -0.319   0.134
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.014   0.172  -0.325   0.346
#>  disinterested.l1     0.093   0.116  -0.138   0.323
#>        excited.l1     0.086   0.188  -0.291   0.448
#>          upset.l1     0.317   0.121   0.079   0.556
#>         strong.l1    -0.067   0.167  -0.387   0.267
#>       stressed.l1     0.150   0.117  -0.079   0.376
#>          steps.l1     0.204   0.110  -0.011   0.418
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.110   0.182  -0.245   0.466
#>  disinterested.l1    -0.023   0.124  -0.260   0.219
#>        excited.l1     0.102   0.200  -0.298   0.501
#>          upset.l1    -0.093   0.132  -0.349   0.167
#>         strong.l1    -0.189   0.181  -0.543   0.164
#>       stressed.l1     0.131   0.124  -0.112   0.374
#>          steps.l1     0.040   0.114  -0.187   0.260
#> ---

# }
```
