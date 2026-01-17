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
#>  interested--disinterested    -0.178   0.100  -0.358   0.026
#>        interested--excited     0.404   0.087   0.225   0.569
#>     disinterested--excited    -0.167   0.103  -0.360   0.043
#>          interested--upset    -0.199   0.105  -0.398   0.008
#>       disinterested--upset    -0.045   0.106  -0.252   0.169
#>             excited--upset    -0.125   0.108  -0.325   0.090
#>         interested--strong     0.313   0.102   0.102   0.503
#>      disinterested--strong     0.104   0.102  -0.097   0.306
#>            excited--strong     0.480   0.082   0.301   0.626
#>              upset--strong     0.103   0.105  -0.112   0.298
#>       interested--stressed     0.279   0.093   0.088   0.451
#>    disinterested--stressed     0.150   0.097  -0.035   0.339
#>          excited--stressed    -0.179   0.101  -0.371   0.025
#>            upset--stressed     0.362   0.094   0.167   0.536
#>           strong--stressed    -0.010   0.104  -0.210   0.191
#>          interested--steps     0.089   0.104  -0.110   0.298
#>       disinterested--steps    -0.085   0.107  -0.283   0.134
#>             excited--steps    -0.009   0.100  -0.202   0.183
#>               upset--steps    -0.049   0.105  -0.250   0.165
#>              strong--steps     0.173   0.104  -0.042   0.359
#>            stressed--steps    -0.023   0.103  -0.228   0.175
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
#>     interested.l1     0.221   0.178  -0.125   0.563
#>  disinterested.l1    -0.051   0.120  -0.280   0.180
#>        excited.l1    -0.080   0.195  -0.471   0.302
#>          upset.l1    -0.154   0.128  -0.412   0.096
#>         strong.l1     0.025   0.173  -0.319   0.364
#>       stressed.l1    -0.020   0.120  -0.260   0.206
#>          steps.l1    -0.155   0.112  -0.373   0.071
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.013   0.177  -0.360   0.336
#>  disinterested.l1    -0.007   0.121  -0.245   0.232
#>        excited.l1    -0.183   0.195  -0.567   0.191
#>          upset.l1     0.259   0.127   0.012   0.508
#>         strong.l1     0.175   0.174  -0.167   0.516
#>       stressed.l1    -0.010   0.121  -0.245   0.227
#>          steps.l1     0.179   0.109  -0.036   0.390
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.178   0.180  -0.173   0.542
#>  disinterested.l1     0.055   0.123  -0.188   0.296
#>        excited.l1     0.004   0.199  -0.376   0.396
#>          upset.l1    -0.094   0.130  -0.346   0.164
#>         strong.l1     0.027   0.175  -0.311   0.371
#>       stressed.l1    -0.032   0.120  -0.271   0.199
#>          steps.l1    -0.210   0.115  -0.441   0.023
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.100   0.172  -0.432   0.240
#>  disinterested.l1    -0.018   0.116  -0.248   0.209
#>        excited.l1     0.053   0.186  -0.302   0.420
#>          upset.l1     0.429   0.126   0.182   0.685
#>         strong.l1     0.050   0.172  -0.296   0.383
#>       stressed.l1    -0.042   0.117  -0.275   0.178
#>          steps.l1     0.148   0.109  -0.071   0.357
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.173   0.181  -0.188   0.523
#>  disinterested.l1     0.047   0.123  -0.204   0.289
#>        excited.l1    -0.083   0.200  -0.472   0.308
#>          upset.l1     0.058   0.131  -0.197   0.322
#>         strong.l1     0.184   0.177  -0.159   0.533
#>       stressed.l1    -0.075   0.123  -0.317   0.162
#>          steps.l1    -0.094   0.115  -0.321   0.128
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.014   0.170  -0.317   0.337
#>  disinterested.l1     0.088   0.118  -0.144   0.317
#>        excited.l1     0.086   0.184  -0.272   0.454
#>          upset.l1     0.318   0.123   0.076   0.559
#>         strong.l1    -0.066   0.172  -0.407   0.277
#>       stressed.l1     0.152   0.116  -0.069   0.379
#>          steps.l1     0.202   0.109  -0.013   0.415
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.110   0.183  -0.260   0.469
#>  disinterested.l1    -0.024   0.125  -0.272   0.217
#>        excited.l1     0.100   0.204  -0.301   0.506
#>          upset.l1    -0.094   0.130  -0.348   0.159
#>         strong.l1    -0.184   0.181  -0.547   0.172
#>       stressed.l1     0.130   0.125  -0.121   0.380
#>          steps.l1     0.040   0.118  -0.190   0.269
#> ---

# }
```
