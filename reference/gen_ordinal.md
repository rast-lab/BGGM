# Generate Ordinal and Binary data

Generate Multivariate Ordinal and Binary data.

## Usage

``` r
gen_ordinal(n, p, levels = 2, cor_mat, empirical = FALSE)
```

## Arguments

- n:

  Number of observations (*n*).

- p:

  Number of variables (*p*).

- levels:

  Number of categories (defaults to 2; binary data).

- cor_mat:

  A *p* by *p* matrix including the true correlation structure.

- empirical:

  Logical. If true, `cor_mat` specifies the empirical not population
  covariance matrix.

## Value

A *n* by *p* data matrix.

## Note

In order to allow users to enjoy the functionality of **BGGM**, we had
to make minor changes to the function `rmvord_naiv` from the `R` package
**orddata** (Leisch et al. 2010) . All rights to, and credit for, the
function `rmvord_naiv` belong to the authors of that package.

This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 2 of the License, or (at your
option) any later version. This program is distributed in the hope that
it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details. A copy of the GNU General
Public License is available online.

## References

Leisch F, Kaiser AWS, Hornik K (2010). *orddata: Generation of
Artificial Ordinal and Binary Data*. R package version 0.1/r4,
<https://R-Forge.R-project.org/projects/orddata/>.

## Examples

``` r
################################
######### example 1 ############
################################

main <-  ptsd_cor1[1:5,1:5]
p <- ncol(main)

pcors <- -(cov2cor(solve(main)) -diag(p))
diag(pcors) <- 1
pcors <- ifelse(abs(pcors) < 0.05, 0, pcors)

inv <-  -pcors
diag(inv) <- 1
cors <- cov2cor( solve(inv))

# example data
Y <- BGGM::gen_ordinal(n = 500, p = 5,
                       levels = 2,
                       cor_mat = cors,
                       empirical = FALSE)



################################
######### example 2 ############
################################
# empirical = TRUE

Y <-  gen_ordinal(n = 500,
                  p = 16,
                  levels = 5,
                  cor_mat = ptsd_cor1,
                  empirical = TRUE)
```
