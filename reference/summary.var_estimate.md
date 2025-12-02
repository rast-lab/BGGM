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
#>  interested--disinterested    -0.180   0.097  -0.360   0.018
#>        interested--excited     0.383   0.091   0.191   0.557
#>     disinterested--excited    -0.173   0.100  -0.368   0.026
#>          interested--upset    -0.194   0.101  -0.377   0.016
#>       disinterested--upset    -0.030   0.102  -0.227   0.170
#>             excited--upset    -0.135   0.109  -0.348   0.080
#>         interested--strong     0.326   0.091   0.135   0.492
#>      disinterested--strong     0.101   0.104  -0.101   0.306
#>            excited--strong     0.494   0.078   0.329   0.637
#>              upset--strong     0.116   0.107  -0.091   0.323
#>       interested--stressed     0.279   0.094   0.093   0.454
#>    disinterested--stressed     0.164   0.098  -0.031   0.353
#>          excited--stressed    -0.178   0.104  -0.380   0.023
#>            upset--stressed     0.355   0.096   0.146   0.531
#>           strong--stressed    -0.011   0.104  -0.225   0.186
#>          interested--steps     0.083   0.102  -0.125   0.273
#>       disinterested--steps    -0.089   0.099  -0.281   0.106
#>             excited--steps    -0.010   0.105  -0.210   0.198
#>               upset--steps    -0.049   0.098  -0.229   0.143
#>              strong--steps     0.167   0.105  -0.041   0.363
#>            stressed--steps    -0.017   0.110  -0.242   0.198
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
#>     interested.l1     0.224   0.179  -0.124   0.579
#>  disinterested.l1    -0.046   0.123  -0.287   0.197
#>        excited.l1    -0.076   0.193  -0.456   0.299
#>          upset.l1    -0.151   0.128  -0.400   0.102
#>         strong.l1     0.025   0.175  -0.310   0.375
#>       stressed.l1    -0.022   0.122  -0.261   0.221
#>          steps.l1    -0.154   0.113  -0.381   0.065
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.014   0.177  -0.349   0.340
#>  disinterested.l1    -0.007   0.120  -0.246   0.226
#>        excited.l1    -0.184   0.196  -0.573   0.196
#>          upset.l1     0.258   0.128  -0.001   0.508
#>         strong.l1     0.173   0.176  -0.168   0.511
#>       stressed.l1    -0.009   0.121  -0.247   0.228
#>          steps.l1     0.181   0.113  -0.046   0.398
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.183   0.180  -0.164   0.550
#>  disinterested.l1     0.058   0.125  -0.182   0.304
#>        excited.l1     0.004   0.197  -0.377   0.384
#>          upset.l1    -0.096   0.131  -0.352   0.165
#>         strong.l1     0.026   0.179  -0.318   0.380
#>       stressed.l1    -0.033   0.126  -0.287   0.207
#>          steps.l1    -0.207   0.116  -0.430   0.022
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.101   0.170  -0.434   0.237
#>  disinterested.l1    -0.017   0.116  -0.245   0.207
#>        excited.l1     0.055   0.187  -0.321   0.423
#>          upset.l1     0.429   0.124   0.190   0.673
#>         strong.l1     0.047   0.171  -0.289   0.383
#>       stressed.l1    -0.044   0.118  -0.281   0.191
#>          steps.l1     0.151   0.107  -0.058   0.359
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.177   0.182  -0.175   0.538
#>  disinterested.l1     0.051   0.125  -0.189   0.301
#>        excited.l1    -0.083   0.203  -0.488   0.308
#>          upset.l1     0.057   0.131  -0.196   0.318
#>         strong.l1     0.185   0.182  -0.166   0.548
#>       stressed.l1    -0.076   0.125  -0.327   0.165
#>          steps.l1    -0.091   0.116  -0.319   0.136
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.016   0.172  -0.320   0.353
#>  disinterested.l1     0.092   0.117  -0.138   0.326
#>        excited.l1     0.090   0.191  -0.292   0.471
#>          upset.l1     0.319   0.126   0.076   0.563
#>         strong.l1    -0.067   0.168  -0.400   0.260
#>       stressed.l1     0.150   0.117  -0.082   0.383
#>          steps.l1     0.204   0.109  -0.012   0.420
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.110   0.184  -0.250   0.471
#>  disinterested.l1    -0.023   0.126  -0.272   0.218
#>        excited.l1     0.101   0.201  -0.277   0.502
#>          upset.l1    -0.090   0.131  -0.351   0.162
#>         strong.l1    -0.182   0.182  -0.541   0.171
#>       stressed.l1     0.132   0.124  -0.113   0.372
#>          steps.l1     0.041   0.114  -0.181   0.268
#> ---

# }
```
