# Summary Method for `ggm_compare_explore` Objects

Summarize the posterior hypothesis probabilities

## Usage

``` r
# S3 method for class 'ggm_compare_explore'
summary(object, col_names = TRUE, ...)
```

## Arguments

- object:

  An object of class `ggm_compare_explore`.

- col_names:

  Logical. Should the summary include the column names (default is
  `TRUE`)? Setting to `FALSE` includes the column numbers (e.g.,
  `1--2`).

- ...:

  Currently ignored.

## Value

An object of class `summary.ggm_compare_explore`

## See also

[`ggm_compare_explore`](https://rast-lab.github.io/BGGM/reference/ggm_compare_explore.md)

## Examples

``` r
# \donttest{
# note: iter = 250 for demonstrative purposes

# data
Y <- bfi[complete.cases(bfi),]

# males and females
Ymale <- subset(Y, gender == 1,
                   select = -c(gender,
                               education))[,1:10]

Yfemale <- subset(Y, gender == 2,
                     select = -c(gender,
                                 education))[,1:10]

##########################
### example 1: ordinal ###
##########################

# fit model
fit <- ggm_compare_explore(Ymale,  Yfemale,
                           type = "ordinal",
                           iter = 250,
                           progress = FALSE)
# summary
summ <- summary(fit)

summ
#> BGGM: Bayesian Gaussian Graphical Models 
#> --- 
#> Type: ordinal 
#> Formula:  
#> Posterior Samples: 250 
#> Observations (n):
#>   Group 1: 735 
#>   Group 2: 1501 
#> Variables (p): 10 
#> Relations: 45 
#> Delta: 2 
#> --- 
#> Call: 
#> ggm_compare_explore(Ymale, Yfemale, type = "ordinal", iter = 250, 
#>     progress = FALSE)
#> --- 
#> Hypotheses:
#> H0: rho_g1 = rho_g2 
#> H1: rho_g1 - rho_g2  = 0
#> --- 
#> 
#>  Relation Post.mean Post.sd Pr.H0 Pr.H1
#>  A1--A2    0.009    0.054   0.957 0.043
#>  A1--A3    0.014    0.053   0.960 0.040
#>  A2--A3   -0.048    0.052   0.937 0.063
#>  A1--A4    0.027    0.055   0.954 0.046
#>  A2--A4   -0.027    0.055   0.953 0.047
#>  A3--A4    0.063    0.056   0.923 0.077
#>  A1--A5   -0.010    0.060   0.956 0.044
#>  A2--A5    0.055    0.060   0.933 0.067
#>  A3--A5    0.082    0.053   0.862 0.138
#>  A4--A5   -0.009    0.062   0.952 0.048
#>  A1--C1    0.038    0.052   0.950 0.050
#>  A2--C1   -0.049    0.058   0.940 0.060
#>  A3--C1    0.065    0.051   0.916 0.084
#>  A4--C1    0.075    0.055   0.900 0.100
#>  A5--C1   -0.030    0.053   0.953 0.047
#>  A1--C2   -0.085    0.056   0.874 0.126
#>  A2--C2   -0.014    0.055   0.958 0.042
#>  A3--C2    0.004    0.054   0.960 0.040
#>  A4--C2    0.041    0.052   0.947 0.053
#>  A5--C2   -0.005    0.057   0.957 0.043
#>  C1--C2    0.042    0.045   0.943 0.057
#>  A1--C3    0.017    0.052   0.958 0.042
#>  A2--C3   -0.021    0.059   0.953 0.047
#>  A3--C3   -0.043    0.054   0.946 0.054
#>  A4--C3    0.065    0.052   0.917 0.083
#>  A5--C3   -0.022    0.054   0.956 0.044
#>  C1--C3   -0.008    0.056   0.956 0.044
#>  C2--C3   -0.023    0.050   0.959 0.041
#>  A1--C4    0.022    0.053   0.956 0.044
#>  A2--C4    0.015    0.054   0.957 0.043
#>  A3--C4    0.122    0.055   0.657 0.343
#>  A4--C4   -0.012    0.060   0.956 0.044
#>  A5--C4   -0.077    0.057   0.898 0.102
#>  C1--C4    0.059    0.055   0.927 0.073
#>  C2--C4    0.004    0.059   0.954 0.046
#>  C3--C4   -0.023    0.055   0.954 0.046
#>  A1--C5   -0.026    0.054   0.955 0.045
#>  A2--C5   -0.055    0.054   0.934 0.066
#>  A3--C5    0.072    0.055   0.909 0.091
#>  A4--C5    0.036    0.055   0.950 0.050
#>  A5--C5   -0.002    0.057   0.957 0.043
#>  C1--C5   -0.069    0.060   0.916 0.084
#>  C2--C5    0.043    0.055   0.945 0.055
#>  C3--C5    0.072    0.048   0.895 0.105
#>  C4--C5    0.037    0.049   0.943 0.057
#> --- 
# }
```
