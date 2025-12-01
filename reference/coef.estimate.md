# Compute Regression Parameters for `estimate` Objects

There is a direct correspondence between the inverse covariance matrix
and multiple regression (Kwan 2014; Stephens 1998) . This readily allows
for converting the GGM parameters to regression coefficients. All data
types are supported.

## Usage

``` r
# S3 method for class 'estimate'
coef(object, iter = NULL, progress = TRUE, ...)
```

## Arguments

- object:

  An Object of class `estimate`

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

- `object` An object of class `estimate` (the fitted model).

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

#########################
### example 1: binary ###
#########################
# data
Y = matrix( rbinom(100, 1, .5), ncol=4)

# fit model
fit <- estimate(Y, type = "binary",
                iter = 250,
                progress = TRUE)
#> BGGM: Posterior Sampling 
#> BGGM: Finished

# summarize the partial correlations
reg <- coef(fit, progress = FALSE)

# summary
summ <- summary(reg)

summ
#> BGGM: Bayesian Gaussian Graphical Models 
#> --- 
#> Type: binary 
#> Formula: ~ 1 
#> --- 
#> Call: 
#> estimate(Y = Y, type = "binary", iter = 250, progress = TRUE)
#> --- 
#> Coefficients: 
#>  
#> 1: 
#>  Node Post.mean Post.sd Cred.lb Cred.ub
#>     2     0.403   0.292  -0.164   0.900
#>     3    -0.196   0.286  -0.712   0.338
#>     4     0.039   0.268  -0.500   0.539
#> 
#> 2: 
#>  Node Post.mean Post.sd Cred.lb Cred.ub
#>     1     0.394   0.294  -0.172   0.911
#>     3    -0.140   0.280  -0.700   0.414
#>     4     0.026   0.301  -0.541   0.678
#> 
#> 3: 
#>  Node Post.mean Post.sd Cred.lb Cred.ub
#>     1    -0.223   0.353  -0.871   0.538
#>     2    -0.191   0.339  -0.822   0.428
#>     4    -0.048   0.311  -0.635   0.579
#> 
#> 4: 
#>  Node Post.mean Post.sd Cred.lb Cred.ub
#>     1     0.056   0.340  -0.577   0.713
#>     2     0.000   0.385  -0.834   0.734
#>     3    -0.053   0.340  -0.715   0.557
#> 
# }
```
