# Model Predictions for `explore` Objects

Model Predictions for `explore` Objects

## Usage

``` r
# S3 method for class 'explore'
predict(
  object,
  newdata = NULL,
  summary = TRUE,
  cred = 0.95,
  iter = NULL,
  progress = TRUE,
  ...
)
```

## Arguments

- object:

  object of class `explore`

- newdata:

  an optional data frame for obtaining predictions (e.g., on test data)

- summary:

  summarize the posterior samples (defaults to `TRUE`).

- cred:

  credible interval used for summarizing

- iter:

  number of posterior samples (defaults to all in the object).

- progress:

  Logical. Should a progress bar be included (defaults to `TRUE`) ?

- ...:

  currently ignored

## Value

`summary = TRUE`: 3D array of dimensions n (observations), 4 (posterior
summary), p (number of nodes). `summary = FALSE`: list containing
predictions for each variable

## Examples

``` r
# \donttest{
# data
Y <- ptsd

# fit model
fit <- explore(Y, iter = 250,
               progress = FALSE)

# predict
pred <- predict(fit,
                progress = FALSE)

# }
```
