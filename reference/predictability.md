# Predictability: Bayesian Variance Explained (R2)

Compute nodewise predictability or Bayesian variance explained (R2
Gelman et al. 2019) . In the context of GGMs, this method was described
in Williams (2018) .

## Usage

``` r
predictability(
  object,
  select = FALSE,
  cred = 0.95,
  BF_cut = 3,
  iter = NULL,
  progress = TRUE,
  ...
)
```

## Arguments

- object:

  object of class `estimate` or `explore`

- select:

  logical. Should the graph be selected ? The default is currently
  `FALSE`.

- cred:

  numeric. credible interval between 0 and 1 (default is 0.95) that is
  used for selecting the graph.

- BF_cut:

  numeric. evidentiary threshold (default is 3).

- iter:

  interger. iterations (posterior samples) used for computing R2.

- progress:

  Logical. Should a progress bar be included (defaults to `TRUE`) ?

- ...:

  currently ignored.

## Value

An object of classes `bayes_R2` and `metric`, including

- `scores` A list containing the posterior samples of R2. The is one
  element

  for each node.

## Note

**Binary and Ordinal Data**:

R2 is computed from the latent data.

**Mixed Data**:

The mixed data approach is somewhat ad-hoc see for example p. 277 in
Hoff (2007) . This is becaue uncertainty in the ranks is not
incorporated, which means that variance explained is computed from the
'empirical' *CDF*.

**Model Selection**:

Currently the default to include all nodes in the model when computing
R2. This can be changed (i.e., `select = TRUE`), which then sets those
edges not detected to zero. This is accomplished by subsetting the
correlation matrix according to each neighborhood of relations.

## References

Gelman A, Goodrich B, Gabry J, Vehtari A (2019). “R-squared for Bayesian
Regression Models.” *American Statistician*, **73**(3), 307–309. ISSN
15372731.  
  
Hoff PD (2007). “Extending the rank likelihood for semiparametric copula
estimation.” *The Annals of Applied Statistics*, **1**(1), 265–283.
[doi:10.1214/07-AOAS107](https://doi.org/10.1214/07-AOAS107) .  
  
Williams DR (2018). “Bayesian Estimation for Gaussian Graphical Models:
Structure Learning, Predictability, and Network Comparisons.” *arXiv*.
[doi:10.31234/OSF.IO/X8DPR](https://doi.org/10.31234/OSF.IO/X8DPR) .

## Examples

``` r
# \donttest{

# data
Y <- ptsd[,1:5]

fit <- estimate(Y, iter = 250, progress = FALSE)

r2 <- predictability(fit, select = TRUE,
                     iter = 250, progress = FALSE)

# summary
r2
#> BGGM: Bayesian Gaussian Graphical Models 
#> --- 
#> Metric: Bayes R2
#> Type: continuous 
#> --- 
#> Estimates:
#> 
#>  Node Post.mean Post.sd Cred.lb Cred.ub
#>    B1     0.444   0.050   0.350   0.549
#>    B2     0.495   0.051   0.403   0.593
#>    B3     0.558   0.047   0.484   0.658
#>    B4     0.505   0.047   0.423   0.604
#>    B5     0.456   0.048   0.355   0.560
# }
```
