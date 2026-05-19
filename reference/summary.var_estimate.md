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
#>  interested--disinterested    -0.181   0.098  -0.365   0.015
#>        interested--excited     0.389   0.095   0.191   0.563
#>     disinterested--excited    -0.186   0.099  -0.376   0.005
#>          interested--upset    -0.217   0.104  -0.397   0.001
#>       disinterested--upset    -0.034   0.108  -0.251   0.180
#>             excited--upset    -0.115   0.099  -0.300   0.090
#>         interested--strong     0.312   0.097   0.114   0.489
#>      disinterested--strong     0.109   0.107  -0.114   0.300
#>            excited--strong     0.496   0.080   0.339   0.644
#>              upset--strong     0.111   0.105  -0.097   0.310
#>       interested--stressed     0.282   0.097   0.085   0.465
#>    disinterested--stressed     0.160   0.101  -0.046   0.353
#>          excited--stressed    -0.166   0.103  -0.365   0.041
#>            upset--stressed     0.356   0.094   0.163   0.536
#>           strong--stressed    -0.015   0.110  -0.230   0.200
#>          interested--steps     0.085   0.105  -0.121   0.298
#>       disinterested--steps    -0.083   0.105  -0.286   0.117
#>             excited--steps    -0.005   0.103  -0.201   0.205
#>               upset--steps    -0.036   0.104  -0.242   0.157
#>              strong--steps     0.174   0.101  -0.029   0.369
#>            stressed--steps    -0.023   0.104  -0.222   0.190
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
#>     interested.l1     0.221   0.177  -0.127   0.566
#>  disinterested.l1    -0.050   0.122  -0.286   0.186
#>        excited.l1    -0.087   0.196  -0.476   0.296
#>          upset.l1    -0.157   0.128  -0.403   0.093
#>         strong.l1     0.033   0.175  -0.318   0.386
#>       stressed.l1    -0.020   0.119  -0.259   0.213
#>          steps.l1    -0.157   0.110  -0.375   0.056
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.017   0.175  -0.363   0.319
#>  disinterested.l1    -0.003   0.118  -0.233   0.230
#>        excited.l1    -0.176   0.194  -0.548   0.206
#>          upset.l1     0.260   0.126   0.011   0.509
#>         strong.l1     0.169   0.175  -0.183   0.509
#>       stressed.l1    -0.011   0.119  -0.234   0.225
#>          steps.l1     0.182   0.112  -0.042   0.401
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.175   0.180  -0.170   0.524
#>  disinterested.l1     0.055   0.124  -0.186   0.300
#>        excited.l1     0.002   0.199  -0.390   0.403
#>          upset.l1    -0.098   0.130  -0.348   0.160
#>         strong.l1     0.031   0.175  -0.313   0.369
#>       stressed.l1    -0.032   0.122  -0.268   0.207
#>          steps.l1    -0.210   0.114  -0.436   0.018
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.100   0.169  -0.435   0.235
#>  disinterested.l1    -0.018   0.116  -0.248   0.209
#>        excited.l1     0.056   0.189  -0.307   0.432
#>          upset.l1     0.430   0.122   0.193   0.667
#>         strong.l1     0.043   0.164  -0.287   0.360
#>       stressed.l1    -0.043   0.114  -0.271   0.183
#>          steps.l1     0.155   0.106  -0.050   0.363
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.171   0.186  -0.190   0.545
#>  disinterested.l1     0.050   0.125  -0.195   0.298
#>        excited.l1    -0.085   0.199  -0.483   0.304
#>          upset.l1     0.054   0.132  -0.211   0.311
#>         strong.l1     0.188   0.179  -0.165   0.546
#>       stressed.l1    -0.073   0.123  -0.317   0.171
#>          steps.l1    -0.092   0.115  -0.322   0.129
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.015   0.173  -0.316   0.358
#>  disinterested.l1     0.087   0.117  -0.135   0.319
#>        excited.l1     0.087   0.186  -0.279   0.440
#>          upset.l1     0.318   0.124   0.077   0.562
#>         strong.l1    -0.069   0.163  -0.393   0.258
#>       stressed.l1     0.152   0.115  -0.074   0.378
#>          steps.l1     0.206   0.107  -0.001   0.419
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.107   0.183  -0.252   0.466
#>  disinterested.l1    -0.025   0.125  -0.270   0.224
#>        excited.l1     0.096   0.202  -0.298   0.491
#>          upset.l1    -0.095   0.130  -0.352   0.157
#>         strong.l1    -0.181   0.181  -0.537   0.168
#>       stressed.l1     0.130   0.122  -0.111   0.364
#>          steps.l1     0.040   0.117  -0.190   0.270
#> ---

# }
```
