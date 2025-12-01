# Package index

## Missing Data

Handle Missing Values.

- [`bggm_missing()`](https://rast-lab.github.io/BGGM/reference/bggm_missing.md)
  : GGM: Missing Data
- [`impute_data()`](https://rast-lab.github.io/BGGM/reference/impute_data.md)
  : Obtain Imputed Datasets

## Estimation Based Methods

‘Estimation’ indicates that the methods to not employ Bayes factor
testing. Rather, the graph is determined with the posterior
distribution. The prior distribtuion has a minimal influence.

- [`estimate()`](https://rast-lab.github.io/BGGM/reference/estimate.md)
  : GGM: Estimation

- [`coef(`*`<estimate>`*`)`](https://rast-lab.github.io/BGGM/reference/coef.estimate.md)
  :

  Compute Regression Parameters for `estimate` Objects

- [`predict(`*`<estimate>`*`)`](https://rast-lab.github.io/BGGM/reference/predict.estimate.md)
  :

  Model Predictions for `estimate` Objects

- [`plot(`*`<summary.estimate>`*`)`](https://rast-lab.github.io/BGGM/reference/plot.summary.estimate.md)
  :

  Plot `summary.estimate` Objects

- [`select(`*`<estimate>`*`)`](https://rast-lab.github.io/BGGM/reference/select.estimate.md)
  :

  Graph Selection for `estimate` Objects

- [`summary(`*`<estimate>`*`)`](https://rast-lab.github.io/BGGM/reference/summary.estimate.md)
  :

  Summary method for `estimate.default` objects

## Exploratory Hypothesis Testing

Bayes factor testing to determine the graph. ‘Exploratory’ reflects that
there is not a specific hypothesis being test.

- [`explore()`](https://rast-lab.github.io/BGGM/reference/explore.md) :
  GGM: Exploratory Hypothesis Testing

- [`coef(`*`<explore>`*`)`](https://rast-lab.github.io/BGGM/reference/coef.explore.md)
  :

  Compute Regression Parameters for `explore` Objects

- [`predict(`*`<explore>`*`)`](https://rast-lab.github.io/BGGM/reference/predict.explore.md)
  :

  Model Predictions for `explore` Objects

- [`plot(`*`<summary.explore>`*`)`](https://rast-lab.github.io/BGGM/reference/plot.summary.explore.md)
  :

  Plot `summary.explore` Objects

- [`plot(`*`<summary.select.explore>`*`)`](https://rast-lab.github.io/BGGM/reference/plot.summary.select.explore.md)
  :

  Plot `summary.select.explore` Objects

- [`select(`*`<explore>`*`)`](https://rast-lab.github.io/BGGM/reference/select.explore.md)
  :

  Graph selection for `explore` Objects

- [`summary(`*`<explore>`*`)`](https://rast-lab.github.io/BGGM/reference/summary.explore.md)
  :

  Summary Method for `explore.default` Objects

- [`summary(`*`<select.explore>`*`)`](https://rast-lab.github.io/BGGM/reference/summary.select.explore.md)
  :

  Summary Method for `select.explore` Objects

## Confirmatory Hypothesis Testing

Test (in)equality constrained hypotheses with the Bayes factor.

- [`confirm()`](https://rast-lab.github.io/BGGM/reference/confirm.md) :
  GGM: Confirmatory Hypothesis Testing

- [`plot(`*`<confirm>`*`)`](https://rast-lab.github.io/BGGM/reference/plot.confirm.md)
  :

  Plot `confirm` objects

## Compare Gaussian Graphical Models

A variety of methods for comparing GGMs.

### Posterior Predictive Check

Compare groups with a posterior predictive check, where the null model
is that the groups are equal. This works with any number of groups.
There is also an option to compare the groups with a user defined
test-statistic.

- [`ggm_compare_ppc()`](https://rast-lab.github.io/BGGM/reference/ggm_compare_ppc.md)
  : GGM Compare: Posterior Predictive Check

- [`plot(`*`<ggm_compare_ppc>`*`)`](https://rast-lab.github.io/BGGM/reference/plot.ggm_compare_ppc.md)
  :

  Plot `ggm_compare_ppc` Objects

### Partial Correlation Differences

Pairwise comparisons for each partial correlation in the respective
models. This can be used for any number of groups. There is also an
analytical solution.

- [`ggm_compare_estimate()`](https://rast-lab.github.io/BGGM/reference/ggm_compare_estimate.md)
  : GGM Compare: Estimate

- [`plot(`*`<summary.ggm_compare_estimate>`*`)`](https://rast-lab.github.io/BGGM/reference/plot.summary.ggm_compare_estimate.md)
  :

  Plot `summary.ggm_compare_estimate` Objects

- [`select(`*`<ggm_compare_estimate>`*`)`](https://rast-lab.github.io/BGGM/reference/select.ggm_compare_estimate.md)
  :

  Graph Selection for `ggm_compare_estimate` Objects

- [`summary(`*`<ggm_compare_estimate>`*`)`](https://rast-lab.github.io/BGGM/reference/summary.ggm_compare_estimate.md)
  :

  Summary method for `ggm_compare_estimate` objects

### Exploratory Hypothesis Testing

Pairwise comparisions with exploratory hypothesis testing. This method
can be used to compare several groups simultaneously.

- [`ggm_compare_explore()`](https://rast-lab.github.io/BGGM/reference/ggm_compare_explore.md)
  : GGM Compare: Exploratory Hypothesis Testing

- [`plot(`*`<summary.ggm_compare_explore>`*`)`](https://rast-lab.github.io/BGGM/reference/plot.summary.ggm_compare_explore.md)
  :

  Plot `summary.ggm_compare_explore` Objects

- [`select(`*`<ggm_compare_explore>`*`)`](https://rast-lab.github.io/BGGM/reference/select.ggm_compare_explore.md)
  :

  Graph selection for `ggm_compare_explore` Objects

- [`summary(`*`<ggm_compare_explore>`*`)`](https://rast-lab.github.io/BGGM/reference/summary.ggm_compare_explore.md)
  :

  Summary Method for `ggm_compare_explore` Objects

### Confirmatory Hypothesis Testing

Test (in)equality constrained hypotheses with the Bayes factor.

- [`ggm_compare_confirm()`](https://rast-lab.github.io/BGGM/reference/ggm_compare_confirm.md)
  : GGM Compare: Confirmatory Hypothesis Testing

- [`plot(`*`<confirm>`*`)`](https://rast-lab.github.io/BGGM/reference/plot.confirm.md)
  :

  Plot `confirm` objects

## Predictability

Bayesian variance explained for each node in the model.

- [`predictability()`](https://rast-lab.github.io/BGGM/reference/predictability.md)
  : Predictability: Bayesian Variance Explained (R2)

- [`plot(`*`<predictability>`*`)`](https://rast-lab.github.io/BGGM/reference/plot.predictability.md)
  :

  Plot `predictability` Objects

- [`summary(`*`<predictability>`*`)`](https://rast-lab.github.io/BGGM/reference/summary.predictability.md)
  :

  Summary Method for `predictability` Objects

## Network Statistics

Compute network statistics from a partial correlation matrix or a
weighted adjacency matrix.

- [`roll_your_own()`](https://rast-lab.github.io/BGGM/reference/roll_your_own.md)
  : Compute Custom Network Statistics

- [`plot(`*`<roll_your_own>`*`)`](https://rast-lab.github.io/BGGM/reference/plot.roll_your_own.md)
  :

  Plot `roll_your_own` Objects

## Partial Correlation Sums

Compute the sum of partial correlations within (one group) or between
(two groups) GGMs. This can be used to compare sums.

- [`pcor_sum()`](https://rast-lab.github.io/BGGM/reference/pcor_sum.md)
  : Partial Correlation Sum

- [`plot(`*`<pcor_sum>`*`)`](https://rast-lab.github.io/BGGM/reference/plot.pcor_sum.md)
  :

  Plot `pcor_sum` Object

## Network Plot

Network plot for the selected graphs. This works with all method for
which there is a selected graph.

- [`plot(`*`<select>`*`)`](https://rast-lab.github.io/BGGM/reference/plot.select.md)
  :

  Network Plot for `select` Objects

## Graphical VAR (vector autoregression)

A variety of methods for time series data. These particular models are
VAR(1) models which are also known as time series chain graphical
models.

### Estimation

‘Estimation’ indicates that the methods to not employ Bayes factor
testing. Rather, the graph is determined with the posterior
distribution. The prior distribtuion has a minimal influence.

- [`var_estimate()`](https://rast-lab.github.io/BGGM/reference/var_estimate.md)
  : VAR: Estimation

- [`select(`*`<var_estimate>`*`)`](https://rast-lab.github.io/BGGM/reference/select.var_estimate.md)
  :

  Graph Selection for `var.estimate` Object

- [`summary(`*`<var_estimate>`*`)`](https://rast-lab.github.io/BGGM/reference/summary.var_estimate.md)
  :

  Summary Method for `var_estimate` Objects

- [`plot(`*`<summary.var_estimate>`*`)`](https://rast-lab.github.io/BGGM/reference/plot.summary.var_estimate.md)
  :

  Plot `summary.var_estimate` Objects

- [`predict(`*`<var_estimate>`*`)`](https://rast-lab.github.io/BGGM/reference/predict.var_estimate.md)
  :

  Model Predictions for `var_estimate` Objects

## Miscellaneous

- [`convergence()`](https://rast-lab.github.io/BGGM/reference/convergence.md)
  : MCMC Convergence

- [`fisher_z_to_r()`](https://rast-lab.github.io/BGGM/reference/fisher_z_to_r.md)
  : Fisher Z Back Transformation

- [`fisher_r_to_z()`](https://rast-lab.github.io/BGGM/reference/fisher_r_to_z.md)
  : Fisher Z Transformation

- [`gen_ordinal()`](https://rast-lab.github.io/BGGM/reference/gen_ordinal.md)
  : Generate Ordinal and Binary data

- [`pcor_to_cor()`](https://rast-lab.github.io/BGGM/reference/pcor_to_cor.md)
  : Compute Correlations from the Partial Correlations

- [`pcor_mat()`](https://rast-lab.github.io/BGGM/reference/pcor_mat.md)
  : Extract the Partial Correlation Matrix

- [`plot_prior()`](https://rast-lab.github.io/BGGM/reference/plot_prior.md)
  : Plot: Prior Distribution

- [`posterior_samples()`](https://rast-lab.github.io/BGGM/reference/posterior_samples.md)
  : Extract Posterior Samples

- [`map()`](https://rast-lab.github.io/BGGM/reference/map.md) : Maximum
  A Posteriori Precision Matrix

- [`regression_summary()`](https://rast-lab.github.io/BGGM/reference/regression_summary.md)
  : Summarary Method for Multivariate or Univarate Regression

- [`summary(`*`<coef>`*`)`](https://rast-lab.github.io/BGGM/reference/summary.coef.md)
  :

  Summarize `coef` Objects

- [`weighted_adj_mat()`](https://rast-lab.github.io/BGGM/reference/weighted_adj_mat.md)
  : Extract the Weighted Adjacency Matrix

- [`zero_order_cors()`](https://rast-lab.github.io/BGGM/reference/zero_order_cors.md)
  : Zero-Order Correlations

## Data

Example datasets and correlation matrices.

- [`asd_ocd`](https://rast-lab.github.io/BGGM/reference/asd_ocd.md) :
  Data: Autism and Obssesive Compulsive Disorder
- [`bfi`](https://rast-lab.github.io/BGGM/reference/bfi.md) : Data: 25
  Personality items representing 5 factors
- [`csws`](https://rast-lab.github.io/BGGM/reference/csws.md) : Data:
  Contingencies of Self-Worth Scale (CSWS)
- [`depression_anxiety_t1`](https://rast-lab.github.io/BGGM/reference/depression_anxiety_t1.md)
  : Data: Depression and Anxiety (Time 1)
- [`depression_anxiety_t2`](https://rast-lab.github.io/BGGM/reference/depression_anxiety_t2.md)
  : Data: Depression and Anxiety (Time 2)
- [`gss`](https://rast-lab.github.io/BGGM/reference/gss.md) : Data: 1994
  General Social Survey
- [`ifit`](https://rast-lab.github.io/BGGM/reference/ifit.md) : Data:
  ifit Intensive Longitudinal Data
- [`iri`](https://rast-lab.github.io/BGGM/reference/iri.md) : Data:
  Interpersonal Reactivity Index (IRI)
- [`ptsd`](https://rast-lab.github.io/BGGM/reference/ptsd.md) : Data:
  Post-Traumatic Stress Disorder
- [`ptsd_cor1`](https://rast-lab.github.io/BGGM/reference/ptsd_cor1.md)
  : Data: Post-Traumatic Stress Disorder (Sample \# 1)
- [`ptsd_cor2`](https://rast-lab.github.io/BGGM/reference/ptsd_cor2.md)
  : Data: Post-Traumatic Stress Disorder (Sample \# 2)
- [`ptsd_cor3`](https://rast-lab.github.io/BGGM/reference/ptsd_cor3.md)
  : Data: Post-Traumatic Stress Disorder (Sample \# 3)
- [`ptsd_cor4`](https://rast-lab.github.io/BGGM/reference/ptsd_cor4.md)
  : Data: Post-Traumatic Stress Disorder (Sample \# 4)
- [`rsa`](https://rast-lab.github.io/BGGM/reference/rsa.md) : Data:
  Resilience Scale of Adults (RSA)
- [`Sachs`](https://rast-lab.github.io/BGGM/reference/Sachs.md) : Data:
  Sachs Network
- [`tas`](https://rast-lab.github.io/BGGM/reference/tas.md) : Data:
  Toronto Alexithymia Scale (TAS)
- [`women_math`](https://rast-lab.github.io/BGGM/reference/women_math.md)
  : Data: Women and Mathematics
