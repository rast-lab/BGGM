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
#>    B1     0.444   0.044   0.365   0.533
#>    B2     0.500   0.045   0.417   0.595
#>    B3     0.554   0.046   0.471   0.649
#>    B4     0.503   0.048   0.418   0.603
#>    B5     0.475   0.046   0.386   0.564

# }
```
