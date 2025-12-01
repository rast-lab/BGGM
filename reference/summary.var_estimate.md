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
#>  interested--disinterested    -0.181   0.099  -0.365   0.018
#>        interested--excited     0.391   0.087   0.208   0.549
#>     disinterested--excited    -0.178   0.102  -0.383   0.015
#>          interested--upset    -0.205   0.100  -0.387   0.003
#>       disinterested--upset    -0.022   0.105  -0.227   0.181
#>             excited--upset    -0.121   0.113  -0.343   0.103
#>         interested--strong     0.314   0.094   0.117   0.487
#>      disinterested--strong     0.110   0.100  -0.091   0.303
#>            excited--strong     0.495   0.074   0.337   0.628
#>              upset--strong     0.102   0.100  -0.096   0.289
#>       interested--stressed     0.281   0.091   0.100   0.453
#>    disinterested--stressed     0.155   0.099  -0.042   0.351
#>          excited--stressed    -0.175   0.101  -0.360   0.035
#>            upset--stressed     0.348   0.095   0.159   0.523
#>           strong--stressed    -0.014   0.106  -0.218   0.193
#>          interested--steps     0.064   0.101  -0.146   0.253
#>       disinterested--steps    -0.076   0.108  -0.291   0.134
#>             excited--steps    -0.005   0.105  -0.207   0.195
#>               upset--steps    -0.045   0.108  -0.258   0.164
#>              strong--steps     0.184   0.101  -0.024   0.375
#>            stressed--steps    -0.012   0.100  -0.202   0.193
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
#>     interested.l1     0.223   0.178  -0.126   0.569
#>  disinterested.l1    -0.049   0.122  -0.290   0.190
#>        excited.l1    -0.080   0.195  -0.456   0.317
#>          upset.l1    -0.152   0.128  -0.405   0.101
#>         strong.l1     0.024   0.173  -0.314   0.364
#>       stressed.l1    -0.021   0.122  -0.257   0.216
#>          steps.l1    -0.151   0.115  -0.380   0.068
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.014   0.179  -0.361   0.340
#>  disinterested.l1    -0.006   0.119  -0.239   0.232
#>        excited.l1    -0.181   0.197  -0.570   0.213
#>          upset.l1     0.256   0.132  -0.003   0.512
#>         strong.l1     0.172   0.178  -0.185   0.520
#>       stressed.l1    -0.007   0.122  -0.244   0.231
#>          steps.l1     0.179   0.113  -0.046   0.406
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.180   0.182  -0.187   0.544
#>  disinterested.l1     0.057   0.125  -0.194   0.300
#>        excited.l1     0.003   0.198  -0.384   0.389
#>          upset.l1    -0.094   0.131  -0.347   0.169
#>         strong.l1     0.025   0.178  -0.321   0.375
#>       stressed.l1    -0.034   0.121  -0.276   0.210
#>          steps.l1    -0.204   0.116  -0.432   0.023
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.101   0.173  -0.441   0.237
#>  disinterested.l1    -0.018   0.118  -0.252   0.216
#>        excited.l1     0.056   0.187  -0.316   0.416
#>          upset.l1     0.427   0.123   0.179   0.672
#>         strong.l1     0.047   0.169  -0.278   0.382
#>       stressed.l1    -0.042   0.115  -0.268   0.186
#>          steps.l1     0.150   0.108  -0.066   0.358
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.176   0.184  -0.195   0.527
#>  disinterested.l1     0.050   0.125  -0.194   0.293
#>        excited.l1    -0.085   0.200  -0.472   0.321
#>          upset.l1     0.058   0.129  -0.199   0.311
#>         strong.l1     0.183   0.176  -0.170   0.524
#>       stressed.l1    -0.076   0.121  -0.311   0.162
#>          steps.l1    -0.088   0.115  -0.313   0.137
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.009   0.171  -0.337   0.346
#>  disinterested.l1     0.090   0.115  -0.134   0.317
#>        excited.l1     0.086   0.188  -0.277   0.449
#>          upset.l1     0.316   0.122   0.076   0.557
#>         strong.l1    -0.063   0.168  -0.400   0.263
#>       stressed.l1     0.153   0.112  -0.073   0.371
#>          steps.l1     0.204   0.107  -0.005   0.418
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.108   0.186  -0.253   0.476
#>  disinterested.l1    -0.022   0.125  -0.267   0.226
#>        excited.l1     0.100   0.203  -0.298   0.502
#>          upset.l1    -0.092   0.133  -0.355   0.167
#>         strong.l1    -0.183   0.181  -0.537   0.168
#>       stressed.l1     0.129   0.125  -0.116   0.369
#>          steps.l1     0.041   0.116  -0.189   0.270
#> ---

# }
```
