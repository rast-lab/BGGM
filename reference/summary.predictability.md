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
#>    B1     0.444   0.046   0.358   0.533
#>    B2     0.507   0.052   0.414   0.622
#>    B3     0.553   0.051   0.460   0.661
#>    B4     0.499   0.048   0.416   0.593
#>    B5     0.463   0.048   0.383   0.565

# }
```
