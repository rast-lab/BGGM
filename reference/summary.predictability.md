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
#>    B1     0.448   0.048   0.365   0.550
#>    B2     0.505   0.049   0.419   0.607
#>    B3     0.552   0.049   0.464   0.645
#>    B4     0.494   0.053   0.407   0.606
#>    B5     0.460   0.048   0.369   0.558

# }
```
