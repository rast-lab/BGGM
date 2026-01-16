library(testthat)
library(BGGM)

# ============================================
# Tests for constrained_posterior()
# ============================================

test_that("constrained_posterior returns correct class", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  # Fit estimate
  fit <- estimate(Y, iter = 100, progress = FALSE)

  # Select graph
  sel <- select(fit)

  # Constrained posterior
  post <- constrained_posterior(
    object = fit,
    adj = sel$adj,
    iter = 50,
    progress = FALSE
  )

  expect_s3_class(post, "BGGM")
  expect_s3_class(post, "constrained")
})

test_that("constrained_posterior returns expected components", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  post <- constrained_posterior(
    object = fit,
    adj = sel$adj,
    iter = 50,
    progress = FALSE
  )

  expect_true("precision_mean" %in% names(post))
  expect_true("pcor_mean" %in% names(post))
  expect_true("precision_samps" %in% names(post))
  expect_true("pcor_samps" %in% names(post))
})

test_that("constrained_posterior matrices have correct dimensions", {
  set.seed(123)
  p <- 5
  Y <- matrix(rnorm(200), 40, p)
  colnames(Y) <- paste0("V", 1:p)

  fit <- estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)
  iter <- 50

  post <- constrained_posterior(
    object = fit,
    adj = sel$adj,
    iter = iter,
    progress = FALSE
  )

  # Check mean matrices
  expect_equal(dim(post$precision_mean), c(p, p))
  expect_equal(dim(post$pcor_mean), c(p, p))

  # Check sample arrays
  expect_equal(dim(post$precision_samps), c(p, p, iter))
  expect_equal(dim(post$pcor_samps), c(p, p, iter))
})

test_that("constrained_posterior errors for non-estimate objects", {
  set.seed(123)
  Y <- matrix(rnorm(100), 20, 5)

  expect_error(
    constrained_posterior(Y, adj = diag(5), iter = 50),
    "object must be of class"
  )
})

test_that("constrained_posterior errors when iter exceeds object iter", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- estimate(Y, iter = 50, progress = FALSE)
  sel <- select(fit)

  expect_error(
    constrained_posterior(object = fit, adj = sel$adj, iter = 100),
    "iter exceeds iter in the object"
  )
})

test_that("constrained_posterior pcor_mean is symmetric", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  post <- constrained_posterior(
    object = fit,
    adj = sel$adj,
    iter = 50,
    progress = FALSE
  )

  expect_equal(post$pcor_mean, t(post$pcor_mean), tolerance = 1e-10)
})

test_that("constrained_posterior works with explore objects", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "two.sided")

  # For explore objects, use Adj_10 as the adjacency matrix
  adj <- sel$Adj_10

  post <- constrained_posterior(
    object = fit,
    adj = adj,
    iter = 50,
    progress = FALSE
  )

  expect_s3_class(post, "constrained")
})

test_that("constrained_posterior respects adjacency constraints", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- estimate(Y, iter = 100, progress = FALSE)

  # Create sparse adjacency matrix (only diagonal)
  adj <- diag(5)

  post <- constrained_posterior(
    object = fit,
    adj = adj,
    iter = 50,
    progress = FALSE
  )

  # Off-diagonal elements of precision_mean should be close to zero
  off_diag <- post$precision_mean[upper.tri(post$precision_mean)]
  expect_true(all(abs(off_diag) < 0.1))
})

test_that("constrained_posterior print method works", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  post <- constrained_posterior(
    object = fit,
    adj = sel$adj,
    iter = 50,
    progress = FALSE
  )

  output <- capture.output(print(post))

  expect_true(length(output) > 0)
  expect_true(any(grepl("BGGM", output)))
})

# ============================================
# Tests for rw_helper (internal function)
# ============================================

test_that("rw_helper returns expected components", {
  set.seed(123)
  p <- 4
  adj <- matrix(c(1, 1, 0, 0,
                  1, 1, 1, 0,
                  0, 1, 1, 1,
                  0, 0, 1, 1), p, p, byrow = TRUE)

  # Create mock posterior parameters
  n_params <- p * (p - 1) / 2 + p
  means_post <- rnorm(n_params)
  cov_post <- diag(n_params) * 0.1

  result <- BGGM:::rw_helper(adj, means_post, cov_post)

  expect_true("mean_post" %in% names(result))
  expect_true("cov_post" %in% names(result))
})
