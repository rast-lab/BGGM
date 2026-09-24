# Changelog

## BGGM 2.2.0

CRAN release: 2026-09-24

#### New features

- **[`select.explore()`](https://rast-lab.github.io/BGGM/reference/select.explore.md)
  gains a `method = "BMA"` option**: Bayesian model averaging as an
  alternative to the hard Bayes factor threshold (`method = "BF_cut"`).
  For each edge the posterior is a spike-and-slab mixture with mass
  `P(H0 | Y)` at zero and the posterior under the alternative otherwise;
  the reported network weights (`pcor_mat_zero`) are the **median of
  this mixture**, computed **exactly** under a normal approximation of
  the posterior of the Fisher-z partial correlation (truncated at 0 for
  one-sided hypotheses). The result is therefore deterministic and does
  not require stored posterior draws. The new `prior.prob.H0` argument
  (default 0.5) sets the prior probability of the null. All alternatives
  are supported. For `alternative = "exhaustive"` the mixture has three
  states – a spike at zero (H0), a positive slab (H+), and a negative
  slab (H-) – mixed by the posterior hypothesis probabilities;
  `pos_mat`/`neg_mat`/`null_mat` then classify each edge by the sign of
  the model-averaged median (so an edge is null when the median is 0,
  which can happen even when H0 is not the most probable hypothesis).
  Thanks to Joris Mulder for the suggestion.
- **[`select.explore()`](https://rast-lab.github.io/BGGM/reference/select.explore.md)
  returns edge inclusion probabilities (`incl_prob`)**: a matrix of
  posterior edge inclusion probabilities `q * BF / (q * BF + 1 - q)`
  with prior inclusion probability `q = 1 - prior.prob.H0` (`BF_10` for
  `"two.sided"`, `BF_20` for `"greater"`/`"less"`, and `1 - P(H0 | Y)`
  for `"exhaustive"`).
- **`prior.prob.H0` now affects the selected graph under
  `method = "BF_cut"`**, not only under `method = "BMA"`. An edge is
  selected when its posterior inclusion probability exceeds
  `BF_cut / (BF_cut + 1)`, and is called null when the posterior
  probability of the null hypothesis exceeds that cutoff. With the
  default `prior.prob.H0 = 0.5` this is identical to the previous Bayes
  factor rule (`BF > BF_cut`) for `"two.sided"`, `"greater"` and
  `"less"`. For `alternative = "exhaustive"` the null keeps
  `prior.prob.H0` and the two directional hypotheses split the remainder
  equally (0.5 / 0.25 / 0.25 by default); because `BF_1u + BF_2u = 2`
  the inclusion probability then equals the one of `"two.sided"`, so
  both alternatives select the same edges, with `"exhaustive"` labelling
  each selected edge positive or negative by the larger of the two
  directional probabilities. `alternative = "exhaustive"` with
  `method = "BF_cut"` now also returns `pcor_mat_zero`. Previously
  `"exhaustive"` used fixed equal 1/3 priors and thresholded each
  hypothesis separately, which made `BF_cut = 3` correspond to a
  two-sided Bayes factor of 1.5.
- **`truncnorm` added to `Imports`**: required for the truncated-normal
  draws used in the one-sided BMA alternatives.
- **[`explore()`](https://rast-lab.github.io/BGGM/reference/explore.md)
  no longer samples the prior by default**: the Bayes factors only need
  the prior sd of the Fisher-z partial correlations, which is now
  computed analytically (numerical integration of the marginal prior
  `rho ~ 2 * Beta(delta/2, delta/2) - 1`, independent of `p`) and stored
  as `prior_sd_z`. Draws from the joint prior (`prior_samp`) are
  returned only with the new argument `store_prior_draws = TRUE`
  (default `FALSE`), which halves time and memory use for large networks
  by default.
  [`select.explore()`](https://rast-lab.github.io/BGGM/reference/select.explore.md),
  [`ggm_compare_explore()`](https://rast-lab.github.io/BGGM/reference/ggm_compare_explore.md)
  and
  [`bggm_missing()`](https://rast-lab.github.io/BGGM/reference/bggm_missing.md)
  use `prior_sd_z`; older `explore` objects still work in
  [`select()`](https://rast-lab.github.io/BGGM/reference/select.md).
  Bayes factors change slightly, because the previously sampled prior sd
  was affected by the matrix-F approximation (e.g. about 0.69-0.70
  instead of 0.684 for `prior_sd = 0.5`).
- **[`explore()`](https://rast-lab.github.io/BGGM/reference/explore.md)
  gains `store_post_draws` (default `TRUE`)**: all samplers used by
  [`explore()`](https://rast-lab.github.io/BGGM/reference/explore.md)
  now keep running posterior summaries of the partial correlations
  (`post_samp$pcor_mat`, `pcor_sd`, `z_mean`, `z_sd`). With
  `store_post_draws = FALSE` the `p x p x iter` arrays `post_samp$pcors`
  and `post_samp$fisher_z` are not stored, so memory use no longer grows
  with the number of iterations.
  [`select()`](https://rast-lab.github.io/BGGM/reference/select.md) (all
  methods and alternatives) and
  [`summary()`](https://rdrr.io/r/base/summary.html) work from these
  summaries when the draws are not stored; functions that need the draws
  ([`posterior_samples()`](https://rast-lab.github.io/BGGM/reference/posterior_samples.md),
  [`convergence()`](https://rast-lab.github.io/BGGM/reference/convergence.md),
  [`coef()`](https://rdrr.io/r/stats/coef.html),
  [`pcor_to_cor()`](https://rast-lab.github.io/BGGM/reference/pcor_to_cor.md),
  [`predict()`](https://rdrr.io/r/stats/predict.html),
  [`predictability()`](https://rast-lab.github.io/BGGM/reference/predictability.md),
  [`posterior_predict()`](https://rast-lab.github.io/BGGM/reference/posterior_predict.md),
  [`constrained_posterior()`](https://rast-lab.github.io/BGGM/reference/constrained_posterior.md))
  stop with an informative error, and
  [`bggm_missing()`](https://rast-lab.github.io/BGGM/reference/bggm_missing.md)
  always stores the draws.
- **[`explore()`](https://rast-lab.github.io/BGGM/reference/explore.md)
  gains `burnin` (default 50) and `thin` (default 1)**: after `burnin`
  iterations, `iter` post-burn-in iterations are run and every `thin`-th
  draw is stored; the running posterior summaries use all post-burn-in
  iterations.
  [`explore()`](https://rast-lab.github.io/BGGM/reference/explore.md)
  objects no longer store the burn-in draws: `post_samp$pcors`,
  `fisher_z`, `beta` and `thresh` now contain only the kept draws
  (previously `iter + 50`, including 50 burn-in draws). Downstream
  functions locate the post-burn-in draws with an internal helper, so
  objects created with earlier versions keep working. Objects from
  [`estimate()`](https://rast-lab.github.io/BGGM/reference/estimate.md),
  [`confirm()`](https://rast-lab.github.io/BGGM/reference/confirm.md)
  and the other functions are unchanged.

#### Bug fixes

- **`explore(iter = )` is now the number of post-burn-in ITERATIONS**,
  not the number of draws kept after thinning. Previously
  `explore(iter = i, thin = t)` ran `burnin + i * t` iterations and
  stored `i` draws, so raising `thin` silently multiplied the run time
  by `t`; it now runs `burnin + i` iterations and stores
  `ceiling(i / thin)` of them, returned in the new `n_draws` element.
  `iter` in the fitted object is the iteration count and `n_draws` the
  stored-draw count. With the default `thin = 1` nothing changes.
- **`seed = NULL` no longer re-seeds the RNG**:
  [`explore()`](https://rast-lab.github.io/BGGM/reference/explore.md),
  [`confirm()`](https://rast-lab.github.io/BGGM/reference/confirm.md),
  [`ggm_compare_confirm()`](https://rast-lab.github.io/BGGM/reference/ggm_compare_confirm.md),
  [`ggm_search()`](https://rast-lab.github.io/BGGM/reference/ggm_search.md)
  and
  [`var_estimate()`](https://rast-lab.github.io/BGGM/reference/var_estimate.md)
  called `set.seed(seed)` unconditionally before the guarded
  `if (!is.null(seed)) set.seed(seed)`. With the default `seed = NULL`
  the first call was `set.seed(NULL)`, which re-initialises the RNG from
  the current time and process ID, discarding any seed the user had set
  and making repeated calls on the same data irreproducible even inside
  [`set.seed()`](https://rdrr.io/r/base/Random.html). The unguarded call
  has been removed, so with `seed = NULL` these functions now use the
  ambient RNG stream and are reproducible under
  [`set.seed()`](https://rdrr.io/r/base/Random.html), as
  [`estimate()`](https://rast-lab.github.io/BGGM/reference/estimate.md)
  already was. Results of existing scripts that relied on the implicit
  re-seeding will change.
- **Corrected the exhaustive posterior hypothesis probabilities**: for
  `alternative = "exhaustive"`, the positive/negative Bayes factors were
  referenced to the null model `H0` instead of the unrestricted model
  `Hu`, so each carried an extra factor of the two-sided Bayes factor
  and double-counted the two-sided evidence. All three Bayes factors are
  now referenced to `Hu` per Eq. 9 of Williams & Mulder (2019). Affects
  `post_prob` and the derived `null_mat`/`pos_mat`/`neg_mat` under both
  `method = "BF_cut"` and `method = "BMA"`.
- **Corrected the prior sd in the `"greater"`, `"less"` and
  `"exhaustive"` selection branches of
  [`select.explore()`](https://rast-lab.github.io/BGGM/reference/select.explore.md)**:
  the `"greater"` branch averaged a hardcoded 3x3 prior mask
  (`upper.tri(diag(3))`), giving wrong Bayes factors whenever the number
  of variables was not 3, and the `"less"`/`"exhaustive"` branches
  averaged the (near-zero) matrix diagonal into the prior standard
  deviation. All branches now use the off-diagonal (edge) prior sd,
  matching the `"two.sided"` branch.
- **Fixed a crash in [`summary()`](https://rdrr.io/r/base/summary.html)
  for
  [`select.explore()`](https://rast-lab.github.io/BGGM/reference/select.explore.md)
  with `alternative = "less"`**: the `"less"` summary branch produced an
  empty `Relation` column (`mat_names[upper.tri(mat_names)]` on an
  already-flattened vector), causing a “differing number of rows” error.
  The greater/less summaries now share one correct branch.
- **[`select.explore()`](https://rast-lab.github.io/BGGM/reference/select.explore.md)
  now uses all post-burn-in draws** for the posterior mean/sd
  (previously the last 50 stored draws were dropped), and no longer
  returns `NA` on the diagonal of the exhaustive
  `null_mat`/`pos_mat`/`neg_mat` (where the posterior sd is 0).
- **Fixed
  [`bggm_missing()`](https://rast-lab.github.io/BGGM/reference/bggm_missing.md)
  dropping the wrong column with `mice` \>= 3.17.0**
  ([\#2](https://github.com/rast-lab/BGGM/issues/2)):
  [`bggm_missing()`](https://rast-lab.github.io/BGGM/reference/bggm_missing.md)
  removed the `.id` column from `mice::complete(action = "long")` by
  position (column 2), which was correct only before `mice` 3.17.0.
  Since `mice` 3.17.0 places `.imp`/`.id` in the last two columns, this
  stripped a real data column and let `.id` leak into the model,
  corrupting the data passed to
  [`estimate()`](https://rast-lab.github.io/BGGM/reference/estimate.md)/
  [`explore()`](https://rast-lab.github.io/BGGM/reference/explore.md).
  The column is now removed by name, which is robust to `mice`’s column
  order. Reported by [@LilyTeesson](https://github.com/LilyTeesson).
- **[`bggm_missing()`](https://rast-lab.github.io/BGGM/reference/bggm_missing.md)
  now pools the posterior draws of the imputed data sets correctly**: it
  stacked all draws of each fit, including their 50 burn-in draws, and
  set `iter` to `iter * m + 50`, so the methods that use draws
  `51:(iter + 50)` included the burn-in draws of imputations 2 to m and
  indexed beyond the stored draws; summaries came from the first
  imputation only. The pooled object now contains the 50 burn-in draws
  of the first fit followed by the post-burn-in draws of all fits, with
  `iter = iter * m`. `beta` and `thresh` are pooled whenever present,
  and `pcor_mat` (and, for `explore`, the posterior summaries) are
  recomputed from the pooled draws. Also works for `m = 1`.
- **[`explore()`](https://rast-lab.github.io/BGGM/reference/explore.md)
  for large or n \< p networks**: the starting value is regularized
  (`solve(cov(Y) + 0.1 I)`), and the matrix-F prior now uses
  `epsilon = min(0.01, 1 / (10 p))` instead of a fixed 0.01, so that
  `nu = 1 / epsilon` stays well above `p - 1` (required for a proper
  prior, Williams & Mulder, 2020). Results change slightly for `p > 10`.
- **C++ samplers no longer truncate the matrix-F degrees of freedom** to
  integers (non-integer `delta` from `prior_sd`).
- **Fixed `rref_ei` not found error**: Added `simple_rref()` function in
  `helpers.R` to replace commented-out
  [`pracma::rref()`](https://rdrr.io/pkg/pracma/man/rref.html) call.
  This fixes a crash in `create_matrices()` when validating constraint
  matrices for hypothesis testing with multiple groups.
- **Fixed
  [`ggm_search()`](https://rast-lab.github.io/BGGM/reference/ggm_search.md)
  crash**: The C++ `search` function now handles edge cases where the
  adjacency matrix becomes all zeros or all ones, preventing “sample
  more elements than in x” errors.
- **Fixed
  [`qplot()`](https://ggplot2.tidyverse.org/reference/qplot.html)
  deprecation warning**: Replaced deprecated
  [`qplot()`](https://ggplot2.tidyverse.org/reference/qplot.html) with
  [`ggplot()`](https://ggplot2.tidyverse.org/reference/ggplot.html) +
  [`geom_density()`](https://ggplot2.tidyverse.org/reference/geom_density.html)
  in
  [`plot_prior()`](https://rast-lab.github.io/BGGM/reference/plot_prior.md).
- **Fixed
  [`ggm_search()`](https://rast-lab.github.io/BGGM/reference/ggm_search.md)
  proposal-set shadowing bug and BIC creep**: The C++
  [`search()`](https://rdrr.io/r/base/search.html) function had a
  variable-shadowing bug where `zeros` and `nonzeros` (pools of
  candidate edges) were re-declared instead of re-assigned after
  accepting a move, freezing them at the starting graph’s configuration
  for the entire run. This caused systematic BIC drift under
  probabilistic acceptance and made reversing bad moves impossible.
  [`ggm_search()`](https://rast-lab.github.io/BGGM/reference/ggm_search.md)
  now implements a proper Metropolis-Hastings sampler with a birth-death
  Hastings correction as the default (`probabilistic = TRUE`); the
  greedy deterministic hill-climb is still available via
  `probabilistic = FALSE`. The `burn_in` parameter is restored and now
  actually applied to discard pre-burn-in samples before computing the
  Bayesian Model Averaging (BMA) solution (probabilistic search only).
  Note: the greedy hill-climb was found to be nearly non-functional on
  realistic test cases, typically accepting only a single edge flip out
  of thousands of attempts.

#### Performance

- **Removed unused posterior arrays from the C++ samplers used by
  [`explore()`](https://rast-lab.github.io/BGGM/reference/explore.md)**
  (also used by
  [`estimate()`](https://rast-lab.github.io/BGGM/reference/estimate.md),
  [`confirm()`](https://rast-lab.github.io/BGGM/reference/confirm.md)
  and the `ggm_compare_*()` functions) and by
  [`var_estimate()`](https://rast-lab.github.io/BGGM/reference/var_estimate.md):
  `Theta_mcmc`, `cors_mcmc` and `Sigma_mcmc` (each p x p x iter) were
  allocated but never filled or returned in `Theta_continuous`,
  `sample_prior`, `mv_continuous`, `mv_binary`, `mv_ordinal_albert`,
  `copula` and `var`; `copula` also allocated an unused n x p x iter
  array of latent data. This reduces memory use substantially for large
  networks (e.g. `mv_continuous`, used with `formula`, allocated five p
  x p x iter arrays and now two).
- **`missing_copula` (mixed data with `impute = TRUE`) no longer returns
  `post_samp$Y_collect`**: this n x p x iter array was never filled (all
  zeros) and was not used anywhere in BGGM.

#### Maintenance

- Added `rlang` to Imports for proper use of `.data` pronoun in ggplot2
  aesthetics.
- Added unit tests for multiple functions.
- **Refactored `bggm_fast.cpp` for clarity and efficiency**: Removed two
  functions unreachable from R (`mv_ordinal_cowles`, `trunc_mvn`; ~390
  lines). Extracted `cov_to_cor()` and `conditional_normal_params()`
  helper functions, eliminating 17 duplicated blocks and a matrix
  inversion that was being computed twice per variable per MCMC
  iteration across 9 sampler functions. Removed dead commented-out debug
  code. No behavior change; file reduced from 3199 to ~2850 lines.
- **Fixed the
  [`ggm_search()`](https://rast-lab.github.io/BGGM/reference/ggm_search.md)
  test suite and added regression tests**: Corrected tests that were
  silently exercising non-existent parameters (`bma = TRUE` instead of
  `bma_mean`, a `start =` argument that doesn’t exist, `gamma =` instead
  of `prior_prob`, `prior_sd =` which isn’t a parameter at all) and a
  `plot.ggm_search()` test that could never fail since no such method
  exists. Rewrote with correct parameter names and assertions on the
  returned object’s fields, and added regression tests tied directly to
  the bugs above (probabilistic search must accept a healthy fraction of
  proposals;
  [`bma_posterior()`](https://rast-lab.github.io/BGGM/reference/bma_posterior.md)
  must not error on the new default’s output). Added
  `tests/testthat/test-search_cpp.R` exercising the C++
  [`search()`](https://rdrr.io/r/base/search.html)/`bic_fast()`/`hft_algorithm()`
  functions directly, including a check for systematic BIC drift across
  a long run.

## BGGM 2.1.6

CRAN release: 2025-12-02

#### Major changes to ordinal sampler

- **Stan-style latent centering for ordinal models**: Both the Albert
  and Cowles ordinals samplers have been refactored to improve numerical
  stability and mixing in the presence of skewed or ceiling/floor
  ordinal items:
  - Thresholds are now initialized from the empirical category
    frequencies of each variable (i.e., cumulative proportions mapped to
    the probit scale), rather than arbitrary or equally spaced
    cut-points.
  - Latent variable draws (Z) are initialized at the expected value of
    the truncated normal (conditional mean), rather than uniform draws
    across the truncation interval. This ensures that ceiling or floor
    items start in the correct region of the latent space.
  - After each update/sweep of a latent column (Z_j), the column is
    recentered (mean ≈ 0) and the corresponding thresholds are shifted
    by the same offset, preserving the likelihood but aligning the
    latent origin. This mirrors the identification scheme used in Stan’s
    ordered-probit/ordered-logit models (latent mean fixed at 0,
    thresholds floating).
  - Probit‐scale bounds (±8 on the standard normal scale) have been
    introduced to cap semi-infinite truncation regions (e.g.,
    ((\_{K-1},∞))) to avoid numerical overflow and improve stability for
    extreme category distributions.

#### Bug fixes & improvements

- Synchronized threshold matrices in the Cowles sampler: the
  `current_thresh`, `candidate_thresh`, `thresh_mat`, `c_thresh_mat`,
  and `thresh_mcmc` now begin from the same baseline after
  initialization to avoid drift in Metropolis proposals.
- Improved sampler performance for ordinal variables with heavy tails or
  heavy ceiling/floor effects. This should lead to reduced shrinkage of
  partial-correlations toward zero when items are highly skewed.

#### Compatibility notes

- The statistical model remains unchanged: you still get the same
  posterior for latent precision/correlation matrices. The changes are
  purely in parameterisation and initialization of the latent (Z) and
  threshold variables.

## BGGM 2.1.5

CRAN release: 2024-12-22

Removed NPM library to avoid CRAN compiler errors

## BGGM 2.1.4

CRAN release: 2024-12-13

### Bug Fixes and Improvements

#### C++ `search` Function

1.  **Improved Initialization**
    - The initial adjacency matrix now takes `start_adj` (the maximum
      likelihood solution) as the starting point to avoid inefficient
      sampling.
2.  **Adaptive Sampling**
    - The sampling of `zeros` and `nonzeros` is now adapted to the
      **newly accepted adjacency matrix** (`adj_s`) rather than the
      static `start_adj`.
3.  **Efficiency Enhancements**
    - Skip updates on rejection since the graph remains unchanged.  
    - Use `find_ids(adj_mat)` instead of `find_ids(start_adj)` to ensure
      edge modifications are correctly tracked after acceptance.

### Minor Changes:

- Example in bggm_missing.R reintroduced; Was removed due to
  irreproducible CRAN error.

- Addressed CRAN check error when building vignettes (removed example in
  bggm_missing.R – will put it back once it’s accepted to CRAN).

- Downgraded required R version to 4.0.0

## BGGM 2.1.3

CRAN release: 2024-07-05

- Replaced dprecated armadillo function `conv_to<>::from` with
  `as_scalar`
- `prior_sd`: Adjusted computation of delta. Also, changed default value
  for estimation: sqrt(1/3) resulting in delta = 2. For model testing
  default is more tight, at `sigma_sd` = 0.5, resulting in delta = 3.
- `prior_sd` is now limited to range 0 – sqrt(1/2)

## BGGM 2.1.2

CRAN release: 2024-06-22

- The prior_sd (or rho_sd in var_estimate() ) is limited to ranges
  between 0 and sqrt(1/8). These values ensure that delta does not go
  below 1.
- *Critical*: select() did not return partial correlations, but Fisher-z
  values in summary(). Fisher values are transformed back to correlation
  metric. This fixes [\#90](https://github.com/rast-lab/BGGM/issues/90),
  see
  [changes](https://github.com/donaldRwilliams/BGGM/commit/a264c440006069e5f171494d9618bae57f4d6566).
- Upgraded deprecated ggplot guides() argument
- Resolved non positive definite initialization matrix in wishrnd() in
  copula models when NA’s are present in observed variables (fixes
  [\#89](https://github.com/rast-lab/BGGM/issues/89)). See changes
  [here](https://github.com/donaldRwilliams/BGGM/commit/d57a5ebabd665907622a1c635ca32b5c6c913184)

## BGGM 2.1.1

CRAN release: 2024-02-23

BFpack dependency error fixed.

## BGGM 2.0.1

CRAN release: 2020-07-23

This version of BGGM included changes based on the JOSS reviews: see
[here](https://github.com/openjournals/joss-reviews/issues/2111) for the
overview and
[here](https://github.com/donaldRwilliams/BGGM/issues?q=is%3Aissue+is%3Aclosed)
for specific issues.

## BGGM 2.0.0

CRAN release: 2020-05-31

**BGGM** was almost completely rewritten for version `2.0.0`. This was
due to adding support for binary, ordinal, and mixed data, which
required that the methods be written in `c ++`. Unfortunately, as a
result, lots of code from version `1.0.0` is broken.

### Added features

- Full support for binary, ordinal, and mixed data. This is implemented
  with the argument `type`

- `roll_your_own`: compute custom network statistics from a weighted
  adjacency matrix or a partial correlation matrix

- `pcor_to_cor`: convert the sampled partial correlation matrices into
  correlation matrices.

- `zero_order_cors`: compute zero order correlations

- `convergence`: acf and trace plots

- `posterior_samples`: extract posterior samples

- `regression_summary`: summarize multivariate regression

- `pcor_sum`: Compute and compare partial correlation sums

- `weighted_adj_mat`: Extract the Weighted Adjacency Matrix

- `pcor_mat`: Extract the Partial Correlation Matrix

- Five additional data sets were added.

### Extensions

- `ggm_compare_ppc`: added option for custom network statistics

- Added option to control for variables with `formula`

- A progress bar was added to many functions

## BGGM 1.0.0

CRAN release: 2020-02-06

Initial CRAN release
