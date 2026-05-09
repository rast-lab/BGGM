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
#>     2     0.309   0.321  -0.352   0.870
#>     3    -0.268   0.292  -0.780   0.414
#>     4    -0.042   0.278  -0.527   0.482
#> 
#> 2: 
#>  Node Post.mean Post.sd Cred.lb Cred.ub
#>     1     0.317   0.331  -0.335   0.901
#>     3    -0.109   0.299  -0.665   0.470
#>     4    -0.007   0.315  -0.505   0.636
#> 
#> 3: 
#>  Node Post.mean Post.sd Cred.lb Cred.ub
#>     1    -0.299   0.353  -0.901   0.608
#>     2    -0.124   0.348  -0.816   0.510
#>     4    -0.094   0.299  -0.627   0.498
#> 
#> 4: 
#>  Node Post.mean Post.sd Cred.lb Cred.ub
#>     1    -0.041   0.349  -0.601   0.687
#>     2    -0.037   0.387  -0.757   0.742
#>     3    -0.100   0.326  -0.775   0.519
#> 
# }
```
