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
#>  interested--disinterested    -0.174   0.102  -0.361   0.038
#>        interested--excited     0.376   0.086   0.205   0.536
#>     disinterested--excited    -0.177   0.099  -0.364   0.021
#>          interested--upset    -0.204   0.095  -0.396  -0.023
#>       disinterested--upset    -0.023   0.097  -0.216   0.173
#>             excited--upset    -0.137   0.103  -0.336   0.063
#>         interested--strong     0.333   0.095   0.133   0.507
#>      disinterested--strong     0.097   0.102  -0.113   0.288
#>            excited--strong     0.494   0.077   0.343   0.635
#>              upset--strong     0.118   0.103  -0.083   0.311
#>       interested--stressed     0.271   0.103   0.065   0.464
#>    disinterested--stressed     0.151   0.101  -0.043   0.341
#>          excited--stressed    -0.159   0.105  -0.354   0.054
#>            upset--stressed     0.357   0.092   0.156   0.518
#>           strong--stressed    -0.008   0.103  -0.210   0.194
#>          interested--steps     0.083   0.108  -0.138   0.288
#>       disinterested--steps    -0.080   0.106  -0.285   0.125
#>             excited--steps    -0.019   0.107  -0.222   0.197
#>               upset--steps    -0.044   0.102  -0.244   0.148
#>              strong--steps     0.184   0.097  -0.004   0.371
#>            stressed--steps    -0.023   0.102  -0.230   0.172
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
#>     interested.l1     0.222   0.175  -0.126   0.568
#>  disinterested.l1    -0.050   0.121  -0.291   0.184
#>        excited.l1    -0.081   0.193  -0.464   0.291
#>          upset.l1    -0.155   0.129  -0.406   0.101
#>         strong.l1     0.024   0.173  -0.309   0.371
#>       stressed.l1    -0.018   0.122  -0.258   0.217
#>          steps.l1    -0.156   0.113  -0.377   0.067
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.016   0.180  -0.372   0.342
#>  disinterested.l1    -0.006   0.120  -0.242   0.229
#>        excited.l1    -0.181   0.193  -0.558   0.193
#>          upset.l1     0.257   0.130  -0.001   0.510
#>         strong.l1     0.173   0.177  -0.176   0.519
#>       stressed.l1    -0.007   0.122  -0.250   0.234
#>          steps.l1     0.182   0.114  -0.043   0.402
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.180   0.180  -0.177   0.530
#>  disinterested.l1     0.055   0.124  -0.189   0.297
#>        excited.l1     0.002   0.197  -0.387   0.387
#>          upset.l1    -0.098   0.133  -0.362   0.160
#>         strong.l1     0.026   0.175  -0.323   0.372
#>       stressed.l1    -0.033   0.125  -0.279   0.210
#>          steps.l1    -0.208   0.115  -0.434   0.017
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.101   0.174  -0.449   0.239
#>  disinterested.l1    -0.019   0.116  -0.245   0.208
#>        excited.l1     0.054   0.188  -0.316   0.418
#>          upset.l1     0.431   0.125   0.183   0.680
#>         strong.l1     0.048   0.168  -0.286   0.374
#>       stressed.l1    -0.045   0.117  -0.277   0.179
#>          steps.l1     0.152   0.107  -0.051   0.367
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.177   0.183  -0.177   0.536
#>  disinterested.l1     0.051   0.124  -0.189   0.297
#>        excited.l1    -0.086   0.201  -0.475   0.308
#>          upset.l1     0.054   0.133  -0.211   0.323
#>         strong.l1     0.184   0.180  -0.180   0.537
#>       stressed.l1    -0.076   0.125  -0.322   0.169
#>          steps.l1    -0.090   0.118  -0.320   0.146
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.014   0.171  -0.318   0.341
#>  disinterested.l1     0.089   0.117  -0.133   0.323
#>        excited.l1     0.086   0.186  -0.272   0.456
#>          upset.l1     0.316   0.123   0.069   0.563
#>         strong.l1    -0.065   0.167  -0.390   0.268
#>       stressed.l1     0.152   0.116  -0.073   0.391
#>          steps.l1     0.205   0.108  -0.008   0.413
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.112   0.179  -0.242   0.459
#>  disinterested.l1    -0.019   0.126  -0.266   0.226
#>        excited.l1     0.103   0.197  -0.287   0.483
#>          upset.l1    -0.092   0.130  -0.349   0.161
#>         strong.l1    -0.186   0.182  -0.545   0.170
#>       stressed.l1     0.130   0.124  -0.115   0.373
#>          steps.l1     0.040   0.117  -0.188   0.272
#> ---

# }
```
