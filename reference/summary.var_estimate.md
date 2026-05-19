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
#>  interested--disinterested    -0.187   0.097  -0.360   0.008
#>        interested--excited     0.376   0.093   0.174   0.540
#>     disinterested--excited    -0.172   0.106  -0.379   0.037
#>          interested--upset    -0.203   0.091  -0.379  -0.019
#>       disinterested--upset    -0.044   0.106  -0.249   0.167
#>             excited--upset    -0.125   0.105  -0.320   0.090
#>         interested--strong     0.340   0.097   0.141   0.523
#>      disinterested--strong     0.090   0.108  -0.137   0.294
#>            excited--strong     0.484   0.082   0.318   0.634
#>              upset--strong     0.109   0.105  -0.107   0.309
#>       interested--stressed     0.275   0.093   0.094   0.465
#>    disinterested--stressed     0.165   0.095  -0.020   0.344
#>          excited--stressed    -0.165   0.103  -0.361   0.032
#>            upset--stressed     0.359   0.100   0.152   0.550
#>           strong--stressed    -0.014   0.106  -0.225   0.196
#>          interested--steps     0.074   0.100  -0.128   0.271
#>       disinterested--steps    -0.070   0.104  -0.272   0.132
#>             excited--steps     0.002   0.102  -0.196   0.199
#>               upset--steps    -0.043   0.104  -0.245   0.171
#>              strong--steps     0.171   0.099  -0.024   0.358
#>            stressed--steps    -0.017   0.103  -0.225   0.186
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
#>     interested.l1     0.225   0.179  -0.123   0.580
#>  disinterested.l1    -0.047   0.122  -0.282   0.192
#>        excited.l1    -0.082   0.199  -0.478   0.306
#>          upset.l1    -0.153   0.130  -0.411   0.100
#>         strong.l1     0.029   0.177  -0.307   0.376
#>       stressed.l1    -0.022   0.121  -0.259   0.215
#>          steps.l1    -0.156   0.113  -0.377   0.065
#> ---
#> disinterested 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.014   0.178  -0.364   0.334
#>  disinterested.l1    -0.007   0.123  -0.241   0.238
#>        excited.l1    -0.183   0.196  -0.565   0.200
#>          upset.l1     0.259   0.129   0.003   0.521
#>         strong.l1     0.169   0.178  -0.180   0.514
#>       stressed.l1    -0.009   0.120  -0.244   0.228
#>          steps.l1     0.182   0.112  -0.033   0.406
#> ---
#> excited 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.183   0.182  -0.174   0.541
#>  disinterested.l1     0.058   0.123  -0.179   0.307
#>        excited.l1     0.003   0.198  -0.380   0.401
#>          upset.l1    -0.096   0.132  -0.357   0.168
#>         strong.l1     0.029   0.178  -0.323   0.377
#>       stressed.l1    -0.034   0.121  -0.267   0.205
#>          steps.l1    -0.210   0.113  -0.428   0.007
#> ---
#> upset 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1    -0.103   0.174  -0.443   0.234
#>  disinterested.l1    -0.021   0.117  -0.249   0.212
#>        excited.l1     0.053   0.188  -0.310   0.423
#>          upset.l1     0.427   0.122   0.188   0.661
#>         strong.l1     0.047   0.169  -0.287   0.374
#>       stressed.l1    -0.043   0.114  -0.263   0.180
#>          steps.l1     0.152   0.109  -0.063   0.368
#> ---
#> strong 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.176   0.183  -0.192   0.536
#>  disinterested.l1     0.052   0.123  -0.183   0.290
#>        excited.l1    -0.083   0.199  -0.472   0.319
#>          upset.l1     0.057   0.132  -0.209   0.310
#>         strong.l1     0.185   0.177  -0.161   0.532
#>       stressed.l1    -0.076   0.123  -0.316   0.166
#>          steps.l1    -0.092   0.114  -0.311   0.129
#> ---
#> stressed 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.015   0.173  -0.324   0.350
#>  disinterested.l1     0.089   0.117  -0.142   0.319
#>        excited.l1     0.086   0.188  -0.271   0.452
#>          upset.l1     0.316   0.124   0.071   0.563
#>         strong.l1    -0.070   0.170  -0.419   0.253
#>       stressed.l1     0.149   0.115  -0.083   0.377
#>          steps.l1     0.204   0.110  -0.015   0.424
#> ---
#> steps 
#> 
#>          Relation Post.mean Post.sd Cred.lb Cred.ub
#>     interested.l1     0.110   0.187  -0.253   0.476
#>  disinterested.l1    -0.022   0.123  -0.262   0.221
#>        excited.l1     0.101   0.200  -0.289   0.503
#>          upset.l1    -0.089   0.130  -0.343   0.166
#>         strong.l1    -0.182   0.180  -0.536   0.171
#>       stressed.l1     0.130   0.124  -0.110   0.379
#>          steps.l1     0.040   0.114  -0.188   0.265
#> ---

# }
```
