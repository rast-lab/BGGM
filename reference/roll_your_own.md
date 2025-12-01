# Compute Custom Network Statistics

This function allows for computing custom network statistics for
weighted adjacency matrices (partial correlations). The statistics are
computed for each of the sampled matrices, resulting in a distribution.

## Usage

``` r
roll_your_own(
  object,
  FUN,
  iter = NULL,
  select = FALSE,
  cred = 0.95,
  progress = TRUE,
  ...
)
```

## Arguments

- object:

  An object of class `estimate`.

- FUN:

  A custom function for computing the statistic. The first argument must
  be a partial correlation matrix.

- iter:

  Number of iterations (posterior samples; defaults to the number in the
  object).

- select:

  Logical. Should the graph be selected ? The default is currently
  `FALSE`.

- cred:

  Numeric. Credible interval between 0 and 1 (default is 0.95) that is
  used for selecting the graph.

- progress:

  Logical. Should a progress bar be included (defaults to `TRUE`) ?

- ...:

  Arguments passed to the function.

## Value

An object defined by `FUN`.

## Details

The user has complete control of this function. Hence, care must be
taken as to what `FUN` returns and in what format. The function should
return a single number (one for the entire GGM) or a vector (one for
each node). This ensures that the print and
[`plot.roll_your_own`](https://rast-lab.github.io/BGGM/reference/plot.roll_your_own.md)
will work.

When `select = TRUE`, the graph is selected and then the network
statistics are computed based on the weigthed adjacency matrix. This is
accomplished internally by multiplying each of the sampled partial
correlation matrices by the adjacency matrix.

## Examples

``` r
# \donttest{
####################################
###### example 1: assortment #######
####################################
# assortment
library(assortnet)

Y <- BGGM::bfi[,1:10]
membership <- c(rep("a", 5), rep("c", 5))

# fit model
fit <- estimate(Y = Y, iter = 250,
                progress = FALSE)

# membership
membership <- c(rep("a", 5), rep("c", 5))

# define function
f <- function(x,...){
 assortment.discrete(x, ...)$r
}


net_stat <- roll_your_own(object = fit,
                          FUN = f,
                          types = membership,
                          weighted = TRUE,
                          SE = FALSE, M = 1,
                          progress = FALSE)

# print
net_stat
#> BGGM: Bayesian Gaussian Graphical Models 
#> --- 
#> Network Stats: Roll Your Own
#> Posterior Samples: 250 
#> --- 
#> Estimates: 
#> 
#>  Post.mean Post.sd Cred.lb Cred.ub
#>      0.365   0.112   0.144   0.587
#> --- 


############################################
###### example 2: expected influence #######
############################################
# expected influence from this package
library(networktools)
#> Registered S3 method overwritten by 'Hmisc':
#>   method          from
#>   summary.formula ergm
#> Registered S3 methods overwritten by 'networktools':
#>   method         from          
#>   print.bridge   bridgesampling
#>   summary.bridge bridgesampling

# data
Y <- depression

# fit model
fit <- estimate(Y = Y, iter = 250)
#> BGGM: Posterior Sampling 
#> BGGM: Finished

# define function
f <- function(x,...){
     expectedInf(x,...)$step1
}

# compute
net_stat <- roll_your_own(object = fit,
                          FUN = f,
                          progress = FALSE)

#######################################
### example 3: mixed data & bridge ####
#######################################
# bridge from this package
library(networktools)

# data
Y <- ptsd[,1:7]

fit <- estimate(Y,
                type = "mixed",
                iter = 250)
#> BGGM: Posterior Sampling 
#> BGGM: Finished

# clusters
communities <- substring(colnames(Y), 1, 1)

# function is slow
f <- function(x, ...){
 bridge(x, ...)$`Bridge Strength`
}

net_stat <- roll_your_own(fit,
                          FUN = f,
                          select = TRUE,
                          communities = communities,
                          progress = FALSE)

# }
```
