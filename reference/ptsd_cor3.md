# Data: Post-Traumatic Stress Disorder (Sample \# 3)

A correlation matrix that includes 16 variables. The correlation matrix
was estimated from 926 individuals (Fried et al. 2018) .

## Format

A correlation matrix with 16 variables

## Details

- Intrusive Thoughts

- Nightmares

- Flashbacks

- Physiological/psychological reactivity

- Avoidance of thoughts

- Avoidance of situations

- Amnesia

- Disinterest in activities

- Feeling detached

- Emotional numbing

- Foreshortened future

- Sleep problems

- Irritability

- Concentration problems

- Hypervigilance

- Startle response

## References

Fried EI, Eidhof MB, Palic S, Costantini G, Huisman-van Dijk HM,
Bockting CL, Engelhard I, Armour C, Nielsen AB, Karstoft K (2018).
“Replicability and generalizability of posttraumatic stress disorder
(PTSD) networks: a cross-cultural multisite study of PTSD symptoms in
four trauma patient samples.” *Clinical Psychological Science*,
**6**(3), 335–351.

## Examples

``` r
data(ptsd_cor3)
Y <- MASS::mvrnorm(n = 926,
                   mu = rep(0, 16),
                   Sigma = ptsd_cor3,
                   empirical = TRUE)
```
