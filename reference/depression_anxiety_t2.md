# Data: Depression and Anxiety (Time 2)

A data frame containing 403 observations (n = 403) and 16 variables (p =
16) measured on the 4-point likert scale (depression: 9; anxiety: 7).

## Usage

``` r
data("depression_anxiety_t2")
```

## Format

A data frame containing 403 observations (n = 7466) and 16 variables (p
= 16) measured on the 4-point likert scale.

## Details

**Depression**:

- `PHQ1` Little interest or pleasure in doing things?

- `PHQ2` Feeling down, depressed, or hopeless?

- `PHQ3` Trouble falling or staying asleep, or sleeping too much?

- `PHQ4` Feeling tired or having little energy?

- `PHQ5` Poor appetite or overeating?

- `PHQ6` Feeling bad about yourself — or that you are a failure or have
  let yourself or your family down?

- `PHQ7` Trouble concentrating on things, such as reading the newspaper
  or watching television?

- `PHQ8` Moving or speaking so slowly that other people could have
  noticed? Or so fidgety or restless that you have been moving a lot
  more than usual?

- `PHQ9` Thoughts that you would be better off dead, or thoughts of
  hurting yourself in some way?

**Anxiety**

- `GAD1` Feeling nervous, anxious, or on edge

- `GAD2` Not being able to stop or control worrying

- `GAD3` Worrying too much about different things

- `GAD4` Trouble relaxing

- `GAD5` Being so restless that it's hard to sit still

- `GAD6` Becoming easily annoyed or irritable

- `GAD7` Feeling afraid as if something awful might happen

## References

Forbes, M. K., Baillie, A. J., & Schniering, C. A. (2016). A structural
equation modeling analysis of the relationships between
depression,anxiety, and sexual problems over time. The Journal of Sex
Research, 53(8), 942-954.

Forbes, M. K., Wright, A. G., Markon, K. E., & Krueger, R. F. (2019).
Quantifying the reliability and replicability of psychopathology network
characteristics. Multivariate behavioral research, 1-19.

Jones, P. J., Williams, D. R., & McNally, R. J. (2019). Sampling
variability is not nonreplication: a Bayesian reanalysis of Forbes,
Wright, Markon, & Krueger.

## Examples

``` r
data("depression_anxiety_t2")
labels<- c("interest", "down", "sleep",
            "tired", "appetite", "selfest",
           "concen", "psychmtr", "suicid",
           "nervous", "unctrworry", "worrylot",
           "relax", "restless", "irritable", "awful")

```
