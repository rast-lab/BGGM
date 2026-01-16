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
#>  interested--disinterested    -0.178   0.101  -0.379   0.027
#>        interested--excited     0.376   0.087   0.194   0.535
#>     disinterested--excited    -0.183   0.102  -0.378   0.022
#>          interested--upset    -0.207   0.100  -0.400  -0.014
#>       disinterested--upset    -0.033   0.104  -0.235   0.174
#>             excited--upset    -0.132   0.106  -0.328   0.081
#>         interested--strong     0.322   0.095   0.137   0.499
#>      disinterested--strong     0.106   0.100  -0.103   0.291
#>            excited--strong     0.506   0.078   0.341   0.649
#>              upset--strong     0.124   0.112  -0.098   0.332
#>       interested--stressed     0.279   0.097   0.086   0.463
#>    disinterested--stressed     0.160   0.102  -0.045   0.346
#>          excited--stressed    -0.170   0.103  -0.367   0.032
#>            upset--stressed     0.343   0.096   0.135   0.509
#>           strong--stressed    -0.010   0.106  -0.216   0.187
#>          interested--steps     0.088   0.110  -0.123   0.301
#>       disinterested--steps    -0.076   0.101  -0.272   0.117
#>             excited--steps    -0.023   0.114  -0.249   0.196
#>               upset--steps    -0.057   0.103  -0.261   0.145
#>              strong--steps     0.184   0.105  -0.023   0.381
#>            stressed--steps    -0.024   0.105  -0.229   0.177
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
#>     interested.l1     0.221   0.181  -0.142   0.580
#>  disinterested.l1    -0.050   0.127  -0.300   0.193
#>        excited.l1    -0.083   0.199  -0.469   0.310
#>          upset.l1    -0.152   0.132  -0.414   0.111
#>         strong.l1     0.028   0.179  -0.325   0.382
#>       stressed.l1    -0.025   0.121  -0.259   0.212
#>          steps.l1    -0.155   0.112  -0.371   0.065
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.015   0.177  -0.358   0.336
#>  disinterested.l1    -0.003   0.121  -0.240   0.239
#>        excited.l1    -0.181   0.193  -0.566   0.208
#>          upset.l1     0.258   0.127   0.008   0.508
#>         strong.l1     0.173   0.175  -0.166   0.517
#>       stressed.l1    -0.007   0.120  -0.237   0.234
#>          steps.l1     0.182   0.112  -0.043   0.401
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.182   0.182  -0.175   0.546
#>  disinterested.l1     0.056   0.126  -0.192   0.298
#>        excited.l1     0.001   0.200  -0.391   0.402
#>          upset.l1    -0.093   0.132  -0.357   0.167
#>         strong.l1     0.027   0.180  -0.336   0.379
#>       stressed.l1    -0.037   0.123  -0.286   0.195
#>          steps.l1    -0.207   0.116  -0.437   0.024
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.100   0.170  -0.437   0.235
#>  disinterested.l1    -0.017   0.116  -0.243   0.206
#>        excited.l1     0.057   0.188  -0.320   0.426
#>          upset.l1     0.427   0.122   0.184   0.669
#>         strong.l1     0.045   0.167  -0.286   0.368
#>       stressed.l1    -0.041   0.116  -0.263   0.189
#>          steps.l1     0.150   0.109  -0.067   0.367
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.176   0.186  -0.192   0.545
#>  disinterested.l1     0.050   0.127  -0.203   0.300
#>        excited.l1    -0.084   0.203  -0.486   0.303
#>          upset.l1     0.058   0.134  -0.212   0.317
#>         strong.l1     0.184   0.181  -0.170   0.539
#>       stressed.l1    -0.080   0.125  -0.328   0.163
#>          steps.l1    -0.092   0.116  -0.323   0.131
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.011   0.174  -0.335   0.355
#>  disinterested.l1     0.090   0.117  -0.135   0.321
#>        excited.l1     0.093   0.190  -0.278   0.462
#>          upset.l1     0.316   0.123   0.073   0.559
#>         strong.l1    -0.071   0.169  -0.404   0.263
#>       stressed.l1     0.154   0.115  -0.074   0.385
#>          steps.l1     0.205   0.109  -0.006   0.418
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.112   0.185  -0.245   0.477
#>  disinterested.l1    -0.024   0.126  -0.270   0.225
#>        excited.l1     0.096   0.198  -0.289   0.485
#>          upset.l1    -0.091   0.135  -0.349   0.169
#>         strong.l1    -0.181   0.180  -0.538   0.168
#>       stressed.l1     0.126   0.126  -0.120   0.376
#>          steps.l1     0.039   0.117  -0.184   0.264
#> ---

# }
```
