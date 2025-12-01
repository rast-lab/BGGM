# Data: Autism and Obssesive Compulsive Disorder

A correlation matrix with 17 variables in total (autsim: 9; OCD: 8). The
sample size was 213.

## Usage

``` r
data("asd_ocd")
```

## Format

A correlation matrix including 17 variables. These data were measured on
a 4 level likert scale.

## Details

**Autism**:

- `CI` Circumscribed interests

- `UP` Unusual preoccupations

- `RO` Repetitive use of objects or interests in parts of objects

- `CR` Compulsions and/or rituals

- `CI` Unusual sensory interests

- `SM` Complex mannerisms or stereotyped body movements

- `SU` Stereotyped utterances/delayed echolalia

- `NIL` Neologisms and/or idiosyncratic language

- `VR` Verbal rituals

**OCD**

- `CD` Concern with things touched due to dirt/bacteria

- `TB` Thoughts of doing something bad around others

- `CT` Continual thoughts that do not go away

- `HP` Belief that someone/higher power put reoccurring thoughts in
  their head

- `CW` Continual washing

- `CCh` Continual checking CntCheck

- `CC` Continual counting/repeating

- `RD` Repeatedly do things until it feels good or just right

## References

Jones, P. J., Ma, R., & McNally, R. J. (2019). Bridge centrality: A
network approach to understanding comorbidity. Multivariate behavioral
research, 1-15.

Ruzzano, L., Borsboom, D., & Geurts, H. M. (2015). Repetitive behaviors
in autism and obsessive-compulsive disorder: New perspectives from a
network analysis. Journal of Autism and Developmental Disorders, 45(1),
192-202. doi:10.1007/s10803-014-2204-9

## Examples

``` r
data("asd_ocd")

# generate continuous
Y <- MASS::mvrnorm(n = 213,
                   mu = rep(0, 17),
                   Sigma = asd_ocd,
                   empirical = TRUE)

```
