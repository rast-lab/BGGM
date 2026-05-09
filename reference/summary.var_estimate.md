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
#>  interested--disinterested    -0.179   0.101  -0.368   0.023
#>        interested--excited     0.384   0.086   0.199   0.542
#>     disinterested--excited    -0.180   0.097  -0.375   0.007
#>          interested--upset    -0.204   0.099  -0.398  -0.006
#>       disinterested--upset    -0.055   0.104  -0.245   0.166
#>             excited--upset    -0.137   0.103  -0.339   0.058
#>         interested--strong     0.327   0.092   0.131   0.490
#>      disinterested--strong     0.105   0.107  -0.106   0.314
#>            excited--strong     0.486   0.078   0.311   0.626
#>              upset--strong     0.119   0.100  -0.084   0.311
#>       interested--stressed     0.280   0.095   0.078   0.451
#>    disinterested--stressed     0.161   0.100  -0.034   0.353
#>          excited--stressed    -0.174   0.104  -0.382   0.029
#>            upset--stressed     0.348   0.090   0.166   0.513
#>           strong--stressed    -0.019   0.106  -0.242   0.189
#>          interested--steps     0.069   0.100  -0.133   0.256
#>       disinterested--steps    -0.076   0.102  -0.273   0.130
#>             excited--steps    -0.009   0.101  -0.197   0.189
#>               upset--steps    -0.053   0.103  -0.259   0.150
#>              strong--steps     0.187   0.091   0.011   0.355
#>            stressed--steps    -0.002   0.097  -0.183   0.200
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
#>     interested.l1     0.223   0.176  -0.115   0.568
#>  disinterested.l1    -0.048   0.123  -0.297   0.193
#>        excited.l1    -0.080   0.196  -0.461   0.309
#>          upset.l1    -0.154   0.128  -0.405   0.095
#>         strong.l1     0.022   0.176  -0.332   0.366
#>       stressed.l1    -0.020   0.121  -0.262   0.212
#>          steps.l1    -0.152   0.113  -0.376   0.072
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.012   0.178  -0.366   0.344
#>  disinterested.l1    -0.005   0.124  -0.247   0.248
#>        excited.l1    -0.182   0.198  -0.572   0.201
#>          upset.l1     0.259   0.129   0.001   0.508
#>         strong.l1     0.174   0.180  -0.175   0.528
#>       stressed.l1    -0.010   0.120  -0.246   0.224
#>          steps.l1     0.181   0.114  -0.040   0.406
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.178   0.178  -0.169   0.527
#>  disinterested.l1     0.056   0.124  -0.188   0.296
#>        excited.l1     0.005   0.196  -0.370   0.400
#>          upset.l1    -0.099   0.128  -0.352   0.155
#>         strong.l1     0.024   0.178  -0.333   0.371
#>       stressed.l1    -0.031   0.122  -0.282   0.201
#>          steps.l1    -0.208   0.116  -0.430   0.021
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.095   0.169  -0.427   0.239
#>  disinterested.l1    -0.017   0.115  -0.239   0.205
#>        excited.l1     0.051   0.184  -0.310   0.409
#>          upset.l1     0.431   0.123   0.190   0.675
#>         strong.l1     0.046   0.169  -0.280   0.391
#>       stressed.l1    -0.043   0.116  -0.274   0.185
#>          steps.l1     0.151   0.107  -0.055   0.365
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.177   0.179  -0.178   0.539
#>  disinterested.l1     0.049   0.125  -0.195   0.291
#>        excited.l1    -0.081   0.197  -0.470   0.306
#>          upset.l1     0.055   0.131  -0.198   0.309
#>         strong.l1     0.179   0.178  -0.177   0.525
#>       stressed.l1    -0.074   0.124  -0.315   0.161
#>          steps.l1    -0.090   0.115  -0.312   0.136
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.011   0.171  -0.318   0.349
#>  disinterested.l1     0.095   0.116  -0.132   0.324
#>        excited.l1     0.090   0.190  -0.290   0.470
#>          upset.l1     0.315   0.122   0.074   0.557
#>         strong.l1    -0.068   0.169  -0.400   0.264
#>       stressed.l1     0.152   0.113  -0.072   0.376
#>          steps.l1     0.205   0.109  -0.008   0.417
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.112   0.185  -0.254   0.473
#>  disinterested.l1    -0.021   0.127  -0.270   0.228
#>        excited.l1     0.102   0.201  -0.295   0.502
#>          upset.l1    -0.092   0.133  -0.355   0.174
#>         strong.l1    -0.189   0.182  -0.549   0.165
#>       stressed.l1     0.129   0.124  -0.110   0.378
#>          steps.l1     0.041   0.116  -0.182   0.268
#> ---

# }
```
