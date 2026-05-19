# MCMC Convergence

Monitor convergence of the MCMC algorithms.

## Usage

``` r
convergence(object, param = NULL, type = "trace", print_names = FALSE)
```

## Arguments

- object:

  An object of class `estimate` or `explore`

- param:

  Character string. Names of parameters for which to monitor MCMC
  convergence.

- type:

  Character string. Which type of convergence plot ? The current options
  are `trace` (default) and `acf`.

- print_names:

  Logical. Should the parameter names be printed (defaults to `FALSE`)?
  This can be used to first determine the parameter names to specify in
  `type`.

## Value

A list of `ggplot` objects.

## Note

An overview of MCMC diagnostics can be found
[here](https://sbfnk.github.io/mfiidd/sessions/mcmc_diagnostics.html).

## Examples

``` r
# \donttest{
# note: iter = 250 for demonstrative purposes

# data
Y <- ptsd[,1:5]

#########################
###### continuous #######
#########################
fit <- estimate(Y, iter = 250,
                progress = FALSE)

# print names first
convergence(fit, print_names = TRUE)
#> Error in convergence(fit, print_names = TRUE): 'data' not of class 'mids'

# trace plots
convergence(fit, type = "trace",
            param = c("B1--B2", "B1--B3"))[[1]]
#> Error in convergence(fit, type = "trace", param = c("B1--B2", "B1--B3")): 'data' not of class 'mids'

# acf plots
convergence(fit, type = "acf",
            param = c("B1--B2", "B1--B3"))[[1]]
#> Error in convergence(fit, type = "acf", param = c("B1--B2", "B1--B3")): 'data' not of class 'mids'
# }
```
