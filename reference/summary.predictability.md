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
#>    B1     0.441   0.044   0.361   0.530
#>    B2     0.504   0.048   0.413   0.597
#>    B3     0.549   0.048   0.455   0.639
#>    B4     0.500   0.045   0.409   0.585
#>    B5     0.466   0.046   0.383   0.561

# }
```
