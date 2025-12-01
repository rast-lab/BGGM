# Data: 1994 General Social Survey

A data frame containing 1002 rows and 7 variables measured on various
scales, including binary and ordered cateogrical (with varying numbers
of categories). There are also missing values in each variable

- `Inc` Income of the respondent in 1000s of dollars, binned into 21
  ordered categories.

- `DEG` Highest degree ever obtained (none, HS, Associates, Bachelors,
  or Graduate)

- `CHILD` Number of children ever had.

- `PINC` Financial status of respondent's parents when respondent was 16
  (on a 5-point scale).

- `PDEG` Maximum of mother's and father's highest degree

- `PCHILD` Number of siblings of the respondent plus one

- `AGE` Age of the respondent in years.

## Usage

``` r
data("gss")
```

## Format

A data frame containing 1190 observations (n = 1190) and 6 variables (p
= 6) measured on the binary scale (Fowlkes et al. 1988) . The variable
descriptions were copied from section 4, Hoff (2007)

## References

Fowlkes EB, Freeny AE, Landwehr JM (1988). “Evaluating logistic models
for large contingency tables.” *Journal of the American Statistical
Association*, **83**(403), 611–622.  
  
Hoff PD (2007). “Extending the rank likelihood for semiparametric copula
estimation.” *The Annals of Applied Statistics*, **1**(1), 265–283.
[doi:10.1214/07-AOAS107](https://doi.org/10.1214/07-AOAS107) .

## Examples

``` r
data("gss")
```
