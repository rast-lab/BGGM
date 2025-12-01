# Data: ifit Intensive Longitudinal Data

A data frame containing 8 variables and nearly 200 observations. There
are two subjects, each of which provided data every data for over 90
days. Six variables are from the PANAS scale (positive and negative
affect), the daily number of steps, and the subject id.

- `id` Subject id

- `interested`

- `disinterested`

- `excited`

- `upset`

- `strong`

- `stressed`

- `steps` steps recorded by a fit bit

## Usage

``` r
data("ifit")
```

## Format

A data frame containing 197 observations and 8 variables. The data have
been used in (O'Laughlin et al. 2020) and (Williams et al. 2019)

## References

O'Laughlin KD, Liu S, Ferrer E (2020). “Use of Composites in Analysis of
Individual Time Series: Implications for Person-Specific Dynamic
Parameters.” *Multivariate Behavioral Research*, 1–18.  
  
Williams DR, Liu S, Martin SR, Rast P (2019). “Bayesian Multivariate
Mixed-Effects Location Scale Modeling of Longitudinal Relations among
Affective Traits, States, and Physical Activity.” *PsyArXiv*.
[doi:10.31234/osf.io/4kfjp](https://doi.org/10.31234/osf.io/4kfjp) .

## Examples

``` r
data("ifit")
```
