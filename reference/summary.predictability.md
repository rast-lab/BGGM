# Summary Method for `predictability` Objects

Summary Method for `predictability` Objects

## Usage

``` r
# S3 method for class 'predictability'
summary(object, cred = 0.95, ...)
```

## Arguments

- object:

  An object of class `predictability`.

- cred:

  Numeric. The credible interval width for summarizing the posterior
  distributions (defaults to 0.95; must be between 0 and 1).

- ...:

  Currently ignored

## Examples

``` r
# \donttest{
Y <- ptsd[,1:5]

fit <- explore(Y, iter = 250,
               progress = FALSE)

r2 <- predictability(fit, iter = 250,
                     progress = FALSE)

summary(r2)
#> BGGM: Bayesian Gaussian Graphical Models 
#> --- 
#> Metric: Bayes R2
#> Type: continuous 
#> --- 
#> Estimates:
#> 
#>  Node Post.mean Post.sd Cred.lb Cred.ub
#>    B1     0.448   0.046   0.369   0.550
#>    B2     0.503   0.048   0.416   0.600
#>    B3     0.549   0.050   0.459   0.647
#>    B4     0.505   0.047   0.409   0.602
#>    B5     0.456   0.047   0.366   0.542

# }
```
