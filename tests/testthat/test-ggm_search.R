library(testthat)
library(BGGM)

# Test data setup
# Note: ggm_search requires sufficient sample size relative to p
set.seed(123)
n <- 200
p <- 4
test_data <- matrix(rnorm(n * p), ncol = p)
colnames(test_data) <- paste0("V", 1:p)

expected_fields <- c(
  "pcor_adj", "Theta_map", "Theta_bma", "pcor_bma", "adj", "adj_start",
  "probs", "approx_marg_ll", "selected", "BF_start", "adj_path", "acc",
  "S", "n"
)

# Basic functionality tests
test_that("ggm_search works with default parameters", {
  result <- ggm_search(
    test_data,
    iter = 200,
    seed = 1,
    progress = FALSE
  )

  expect_s3_class(result, c("BGGM", "ggm_search"))
  expect_true(is.list(result))
})

test_that("ggm_search returns the documented fields", {
  result <- ggm_search(
    test_data,
    iter = 200,
    seed = 1,
    progress = FALSE
  )

  expect_true(all(expected_fields %in% names(result)))
})

test_that("ggm_search returns a valid p x p adjacency matrix", {
  result <- ggm_search(
    test_data,
    iter = 200,
    seed = 1,
    progress = FALSE
  )

  expect_true(is.matrix(result$adj))
  expect_equal(dim(result$adj), c(p, p))
  expect_true(all(result$adj %in% c(0, 1)))
  expect_true(isSymmetric(result$adj))
  expect_true(all(diag(result$adj) == 1))
})

# probabilistic = TRUE is the default: a genuine Metropolis-Hastings sampler
test_that("probabilistic search (default) explores multiple graphs", {
  result <- ggm_search(
    test_data,
    iter = 500,
    seed = 1,
    progress = FALSE
  )

  # a real MH chain on p = 4 (6 possible edges) should accept a
  # meaningful fraction of proposals, not just one or two
  expect_gt(result$acc, 10)
  expect_equal(length(result$approx_marg_ll), dim(result$adj_path)[3])
})

test_that("probabilistic search does not get stuck (regression for the proposal-set shadowing bug)", {
  result <- ggm_search(
    test_data,
    iter = 1000,
    seed = 42,
    progress = FALSE
  )

  # with the historical shadowing bug, zeros/nonzeros were frozen after
  # the first accepted move, strongly capping how many further moves
  # could ever be accepted; a working sampler should clear this bar
  expect_gt(result$acc / 1000, 0.05)
})

# probabilistic = FALSE: the original deterministic greedy hill-climb
test_that("greedy search (probabilistic = FALSE) returns valid output", {
  result <- ggm_search(
    test_data,
    iter = 200,
    seed = 1,
    progress = FALSE,
    probabilistic = FALSE
  )

  expect_s3_class(result, c("BGGM", "ggm_search"))
  expect_true(is.matrix(result$adj))
  expect_equal(dim(result$adj), c(p, p))
  # greedy only accepts a proposal if BIC improves, so the accepted-move
  # trajectory is monotone non-increasing
  expect_true(all(diff(result$approx_marg_ll) <= 1e-8))
})

test_that("greedy search handles the acc == 0 case without error", {
  # very few iterations makes it plausible that zero proposals are
  # accepted; this must not error, and the BMA fields must stay NULL
  result <- ggm_search(
    test_data,
    iter = 3,
    seed = 1,
    progress = FALSE,
    probabilistic = FALSE
  )

  expect_true(is.matrix(result$adj))
  if (result$acc == 0) {
    expect_null(result$Theta_bma)
    expect_null(result$pcor_bma)
  }
})

# burn_in
test_that("burn_in trims the returned trajectory under probabilistic search", {
  iter <- 400
  burn_in <- 300

  result <- ggm_search(
    test_data,
    iter = iter,
    burn_in = burn_in,
    seed = 1,
    progress = FALSE
  )

  expect_lte(length(result$approx_marg_ll), iter - burn_in)
})

test_that("burn_in = 0 keeps the full trajectory", {
  iter <- 200

  result <- ggm_search(
    test_data,
    iter = iter,
    burn_in = 0,
    seed = 1,
    progress = FALSE
  )

  expect_equal(length(result$approx_marg_ll), iter)
})

test_that("burn_in is not applied under greedy search", {
  iter <- 200

  result <- ggm_search(
    test_data,
    iter = iter,
    burn_in = 150,
    seed = 1,
    progress = FALSE,
    probabilistic = FALSE
  )

  # greedy's BIC trajectory has no transient to burn in; the full
  # (stop_early-trimmed) trajectory should be returned regardless of
  # the burn_in argument
  expect_equal(length(result$approx_marg_ll), dim(result$adj_path)[3])
})

# bma_mean
test_that("bma_mean = TRUE returns a BMA solution when moves are accepted", {
  result <- ggm_search(
    test_data,
    iter = 500,
    bma_mean = TRUE,
    seed = 1,
    progress = FALSE
  )

  skip_if(result$acc == 0, "no proposals accepted, nothing to average")
  expect_true(is.matrix(result$Theta_bma))
  expect_true(is.matrix(result$pcor_bma))
  expect_equal(dim(result$Theta_bma), c(p, p))
})

test_that("bma_mean = FALSE skips the BMA solution", {
  result <- ggm_search(
    test_data,
    iter = 200,
    bma_mean = FALSE,
    seed = 1,
    progress = FALSE
  )

  expect_null(result$Theta_bma)
  expect_null(result$pcor_bma)
})

# prior_prob (the real sparsity-prior parameter; note earlier revisions of
# this test file used a nonexistent `gamma` argument, silently swallowed by
# ggm_search()'s `...` and never actually exercised)
test_that("ggm_search accepts prior_prob", {
  result <- ggm_search(
    test_data,
    prior_prob = 0.7,
    iter = 200,
    seed = 1,
    progress = FALSE
  )

  expect_s3_class(result, c("BGGM", "ggm_search"))
  expect_true(is.matrix(result$adj))
})

# print method
test_that("print.ggm_search works", {
  fit <- ggm_search(
    test_data,
    iter = 100,
    seed = 1,
    progress = FALSE
  )

  expect_output(print(fit))
})

# different sample sizes / number of variables
test_that("ggm_search handles different sample sizes", {
  set.seed(456)
  small_data <- matrix(rnorm(30 * 4), ncol = 4)
  colnames(small_data) <- paste0("V", 1:4)

  result <- ggm_search(
    small_data,
    iter = 100,
    seed = 1,
    progress = FALSE
  )

  expect_s3_class(result, c("BGGM", "ggm_search"))
})

test_that("ggm_search handles different numbers of variables", {
  set.seed(789)
  data_5var <- matrix(rnorm(50 * 5), ncol = 5)
  colnames(data_5var) <- paste0("V", 1:5)

  result <- ggm_search(
    data_5var,
    iter = 100,
    seed = 1,
    progress = FALSE
  )

  expect_s3_class(result, c("BGGM", "ggm_search"))
  expect_equal(dim(result$adj), c(5, 5))
})

# bma_posterior integration -- regression test for the ggm_search() <->
# bma_posterior() length-mismatch bug: probs (computed inside ggm_search
# from the burn-in-trimmed trajectory) must stay consistent with the graph
# count bma_posterior() re-derives from the *returned* approx_marg_ll /
# adj_path, or bma_posterior's sample() call errors with "incorrect number
# of probabilities"
test_that("bma_posterior works with the probabilistic ggm_search default", {
  fit <- ggm_search(
    test_data,
    iter = 500,
    bma_mean = TRUE,
    seed = 1,
    progress = FALSE
  )

  skip_if(fit$acc == 0, "no proposals accepted, nothing to average")

  result <- bma_posterior(fit, iter = 20, progress = FALSE)

  expect_type(result, "list")
  expect_named(result, c("bma_mean", "samples"))
  expect_true(is.matrix(result$bma_mean))
  expect_equal(dim(result$bma_mean), c(p, p))
  expect_equal(dim(result$samples), c(p, p, 20))
})

test_that("bma_posterior works with greedy (probabilistic = FALSE) search", {
  fit <- ggm_search(
    test_data,
    iter = 200,
    bma_mean = TRUE,
    seed = 1,
    progress = FALSE,
    probabilistic = FALSE
  )

  skip_if(fit$acc == 0, "no proposals accepted, nothing to average")

  result <- bma_posterior(fit, iter = 20, progress = FALSE)

  expect_type(result, "list")
  expect_equal(dim(result$bma_mean), c(p, p))
})
