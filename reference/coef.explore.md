# Compute Regression Parameters for `explore` Objects

There is a direct correspondence between the inverse covariance matrix
and multiple regression (Kwan 2014; Stephens 1998) . This readily allows
for converting the GGM parameters to regression coefficients. All data
types are supported.

## Usage

``` r
# S3 method for class 'explore'
coef(object, iter = NULL, progress = TRUE, ...)
```

## Arguments

- object:

  An Object of class `explore`.

- iter:

  Number of iterations (posterior samples; defaults to the number in the
  object).

- progress:

  Logical. Should a progress bar be included (defaults to `TRUE`) ?

- ...:

  Currently ignored.

## Value

An object of class `coef`, containting two lists.

- `betas` A list of length *p*, each containing a *p* - 1 by `iter`
  matrix of posterior samples

- `object` An object of class `explore` (the fitted model).

## References

Kwan CC (2014). “A regression-based interpretation of the inverse of the
sample covariance matrix.” *Spreadsheets in Education*, **7**(1),
4613.  
  
Stephens G (1998). “On the Inverse of the Covariance Matrix in Portfolio
Analysis.” *The Journal of Finance*, **53**(5), 1821–1827.

## Examples

``` r
# \donttest{
# note: iter = 250 for demonstrative purposes

# data
Y <- ptsd[,1:4]

##########################
### example 1: ordinal ###
##########################

# fit model (note + 1, due to zeros)
fit <- explore(Y + 1,
               type = "ordinal",
               iter = 250,
               progress = FALSE,
               seed = 1234)

# summarize the partial correlations
reg <- coef(fit, progress = FALSE)

# summary
summ <- summary(reg)

summ
#> BGGM: Bayesian Gaussian Graphical Models 
#> --- 
#> Type: ordinal 
#> Formula: ~ 1 
#> --- 
#> Call: 
#> explore(Y = Y + 1, type = "ordinal", iter = 250, progress = FALSE, 
#>     seed = 1234)
#> --- 
#> Coefficients: 
#>  
#> B1: 
#>  Node Post.mean Post.sd Cred.lb Cred.ub
#>    B2     0.269   0.109   0.043   0.472
#>    B3     0.113   0.115  -0.105   0.332
#>    B4     0.401   0.084   0.225   0.554
#> 
#> B2: 
#>  Node Post.mean Post.sd Cred.lb Cred.ub
#>    B1     0.248   0.103   0.037   0.441
#>    B3     0.528   0.106   0.279   0.688
#>    B4     0.003   0.105  -0.183   0.200
#> 
#> B3: 
#>  Node Post.mean Post.sd Cred.lb Cred.ub
#>    B1     0.093   0.096  -0.080   0.279
#>    B2     0.467   0.099   0.261   0.653
#>    B4     0.315   0.089   0.154   0.488
#> 
#> B4: 
#>  Node Post.mean Post.sd Cred.lb Cred.ub
#>    B1     0.394   0.092   0.193   0.571
#>    B2     0.001   0.107  -0.187   0.194
#>    B3     0.378   0.104   0.172   0.582
#> 
# }
```
