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
#>  interested--disinterested    -0.171   0.099  -0.367   0.020
#>        interested--excited     0.392   0.089   0.215   0.568
#>     disinterested--excited    -0.178   0.099  -0.367   0.018
#>          interested--upset    -0.200   0.105  -0.396   0.014
#>       disinterested--upset    -0.022   0.100  -0.227   0.161
#>             excited--upset    -0.132   0.104  -0.335   0.069
#>         interested--strong     0.317   0.096   0.118   0.492
#>      disinterested--strong     0.100   0.103  -0.102   0.306
#>            excited--strong     0.489   0.080   0.315   0.625
#>              upset--strong     0.119   0.105  -0.104   0.314
#>       interested--stressed     0.271   0.098   0.074   0.450
#>    disinterested--stressed     0.154   0.101  -0.044   0.350
#>          excited--stressed    -0.161   0.105  -0.364   0.047
#>            upset--stressed     0.345   0.094   0.148   0.514
#>           strong--stressed    -0.016   0.098  -0.204   0.179
#>          interested--steps     0.076   0.106  -0.127   0.298
#>       disinterested--steps    -0.078   0.107  -0.286   0.138
#>             excited--steps     0.002   0.102  -0.196   0.205
#>               upset--steps    -0.038   0.103  -0.239   0.160
#>              strong--steps     0.174   0.102  -0.039   0.367
#>            stressed--steps    -0.037   0.109  -0.242   0.173
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
#>     interested.l1     0.224   0.179  -0.117   0.575
#>  disinterested.l1    -0.049   0.122  -0.281   0.192
#>        excited.l1    -0.077   0.194  -0.467   0.305
#>          upset.l1    -0.153   0.128  -0.407   0.099
#>         strong.l1     0.023   0.177  -0.323   0.371
#>       stressed.l1    -0.020   0.120  -0.253   0.220
#>          steps.l1    -0.155   0.113  -0.376   0.073
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.015   0.177  -0.363   0.327
#>  disinterested.l1    -0.007   0.121  -0.242   0.232
#>        excited.l1    -0.186   0.194  -0.559   0.202
#>          upset.l1     0.256   0.128   0.001   0.502
#>         strong.l1     0.174   0.175  -0.182   0.517
#>       stressed.l1    -0.006   0.122  -0.247   0.230
#>          steps.l1     0.184   0.113  -0.040   0.403
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.182   0.177  -0.164   0.530
#>  disinterested.l1     0.057   0.124  -0.181   0.303
#>        excited.l1     0.011   0.196  -0.372   0.397
#>          upset.l1    -0.093   0.130  -0.341   0.157
#>         strong.l1     0.022   0.180  -0.331   0.372
#>       stressed.l1    -0.032   0.122  -0.270   0.209
#>          steps.l1    -0.209   0.114  -0.427   0.016
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.099   0.171  -0.437   0.234
#>  disinterested.l1    -0.017   0.116  -0.243   0.205
#>        excited.l1     0.055   0.188  -0.315   0.432
#>          upset.l1     0.428   0.123   0.187   0.672
#>         strong.l1     0.041   0.169  -0.298   0.368
#>       stressed.l1    -0.042   0.117  -0.275   0.184
#>          steps.l1     0.154   0.109  -0.061   0.376
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.176   0.181  -0.175   0.537
#>  disinterested.l1     0.050   0.125  -0.192   0.296
#>        excited.l1    -0.078   0.196  -0.458   0.297
#>          upset.l1     0.056   0.131  -0.199   0.316
#>         strong.l1     0.178   0.178  -0.171   0.537
#>       stressed.l1    -0.073   0.124  -0.313   0.174
#>          steps.l1    -0.091   0.114  -0.308   0.127
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.013   0.168  -0.319   0.346
#>  disinterested.l1     0.092   0.114  -0.134   0.309
#>        excited.l1     0.085   0.185  -0.293   0.433
#>          upset.l1     0.313   0.122   0.079   0.556
#>         strong.l1    -0.067   0.168  -0.399   0.256
#>       stressed.l1     0.153   0.116  -0.076   0.383
#>          steps.l1     0.205   0.110  -0.008   0.425
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.111   0.181  -0.248   0.463
#>  disinterested.l1    -0.022   0.125  -0.264   0.224
#>        excited.l1     0.099   0.205  -0.300   0.498
#>          upset.l1    -0.091   0.130  -0.353   0.170
#>         strong.l1    -0.181   0.184  -0.542   0.187
#>       stressed.l1     0.129   0.123  -0.110   0.374
#>          steps.l1     0.039   0.113  -0.185   0.264
#> ---

# }
```
