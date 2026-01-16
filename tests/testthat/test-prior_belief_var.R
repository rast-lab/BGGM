library(testthat)
library(BGGM)

# ============================================
# Tests for prior_belief_var()
# ============================================

test_that("prior_belief_var returns correct class", {
  skip_on_cran()

  set.seed(123)
  # Create time series data
  n <- 100
  p <- 4
  Y <- matrix(rnorm(n * p), n, p)
  colnames(Y) <- paste0("V", 1:p)

  suppressMessages({
    fit <- prior_belief_var(Y, est_ggm = FALSE, progress = FALSE)
  })

  expect_s3_class(fit, "BGGM")
  expect_s3_class(fit, "prior_var")
})

test_that("prior_belief_var without GGM returns expected components", {
  skip_on_cran()

  set.seed(123)
  n <- 100
  p <- 4
  Y <- matrix(rnorm(n * p), n, p)
  colnames(Y) <- paste0("V", 1:p)

  suppressMessages({
    fit <- prior_belief_var(Y, est_ggm = FALSE, progress = FALSE)
  })

  expect_true("adj" %in% names(fit))
  expect_true("post_prob" %in% names(fit))
  expect_true("coef_mat" %in% names(fit))
})

test_that("prior_belief_var with GGM returns expected components", {
  skip_on_cran()

  set.seed(123)
  n <- 100
  p <- 4
  Y <- matrix(rnorm(n * p), n, p)
  colnames(Y) <- paste0("V", 1:p)

  suppressMessages({
    fit <- prior_belief_var(Y, est_ggm = TRUE, progress = FALSE)
  })

  expect_true("adj_temporal" %in% names(fit))
  expect_true("post_prob_temporal" %in% names(fit))
  expect_true("adj_ggm" %in% names(fit))
  expect_true("post_prob_ggm" %in% names(fit))
  expect_true("coef_mat" %in% names(fit))
})

test_that("prior_belief_var adjacency matrix has correct dimensions", {
  skip_on_cran()

  set.seed(123)
  n <- 100
  p <- 4
  Y <- matrix(rnorm(n * p), n, p)
  colnames(Y) <- paste0("V", 1:p)

  suppressMessages({
    fit <- prior_belief_var(Y, est_ggm = FALSE, progress = FALSE)
  })

  expect_equal(dim(fit$adj), c(p, p))
  expect_equal(dim(fit$post_prob), c(p, p))
  expect_equal(dim(fit$coef_mat), c(p, p))
})

test_that("prior_belief_var post_prob is in [0,1]", {
  skip_on_cran()

  set.seed(123)
  n <- 100
  p <- 4
  Y <- matrix(rnorm(n * p), n, p)
  colnames(Y) <- paste0("V", 1:p)

  suppressMessages({
    fit <- prior_belief_var(Y, est_ggm = FALSE, progress = FALSE)
  })

  expect_true(all(fit$post_prob >= 0 & fit$post_prob <= 1))
})

test_that("prior_belief_var adjacency is binary", {
  skip_on_cran()

  set.seed(123)
  n <- 100
  p <- 4
  Y <- matrix(rnorm(n * p), n, p)
  colnames(Y) <- paste0("V", 1:p)

  suppressMessages({
    fit <- prior_belief_var(Y, est_ggm = FALSE, progress = FALSE)
  })

  expect_true(all(fit$adj %in% c(0, 1)))
})

test_that("prior_belief_var respects post_odds_cut", {
  skip_on_cran()

  set.seed(123)
  n <- 100
  p <- 4
  Y <- matrix(rnorm(n * p), n, p)
  colnames(Y) <- paste0("V", 1:p)

  # High cutoff should result in sparser graph
  suppressMessages({
    fit_low <- prior_belief_var(Y, post_odds_cut = 1, est_ggm = FALSE, progress = FALSE)
    fit_high <- prior_belief_var(Y, post_odds_cut = 10, est_ggm = FALSE, progress = FALSE)
  })

  # High cutoff should have fewer or equal edges
  expect_true(sum(fit_high$adj) <= sum(fit_low$adj))
})

test_that("prior_belief_var with custom prior_temporal", {
  skip_on_cran()

  set.seed(123)
  n <- 100
  p <- 4
  Y <- matrix(rnorm(n * p), n, p)
  colnames(Y) <- paste0("V", 1:p)

  # Custom prior odds matrix
  prior_var <- matrix(1, p, p)
  prior_var[1, 2] <- 10  # Strong prior for edge 1->2

  suppressMessages({
    fit <- prior_belief_var(Y, prior_temporal = prior_var, est_ggm = FALSE, progress = FALSE)
  })

  expect_s3_class(fit, "prior_var")
})

test_that("prior_belief_var errors with zero in prior_temporal", {
  skip_on_cran()

  set.seed(123)
  n <- 100
  p <- 4
  Y <- matrix(rnorm(n * p), n, p)

  # Prior with zero
  prior_var <- matrix(1, p, p)
  prior_var[1, 2] <- 0

  expect_error(
    suppressMessages(prior_belief_var(Y, prior_temporal = prior_var, progress = FALSE)),
    "zeros are not allowed"
  )
})

test_that("prior_belief_var errors with zero in prior_ggm", {
  skip_on_cran()

  set.seed(123)
  n <- 100
  p <- 4
  Y <- matrix(rnorm(n * p), n, p)

  # Prior with zero
  prior_ggm <- matrix(1, p, p)
  prior_ggm[1, 2] <- 0

  expect_error(
    suppressMessages(prior_belief_var(Y, prior_ggm = prior_ggm, est_ggm = TRUE, progress = FALSE)),
    "zeros are not allowed"
  )
})

test_that("prior_belief_var print method works", {
  skip_on_cran()

  set.seed(123)
  n <- 100
  p <- 4
  Y <- matrix(rnorm(n * p), n, p)
  colnames(Y) <- paste0("V", 1:p)

  suppressMessages({
    fit <- prior_belief_var(Y, est_ggm = FALSE, progress = FALSE)
  })

  output <- capture.output(print(fit))

  expect_true(length(output) > 0)
  expect_true(any(grepl("BGGM", output)))
})

test_that("prior_belief_var handles data frame input", {
  skip_on_cran()

  set.seed(123)
  n <- 100
  p <- 4
  Y <- data.frame(matrix(rnorm(n * p), n, p))
  colnames(Y) <- paste0("V", 1:p)

  suppressMessages({
    fit <- prior_belief_var(Y, est_ggm = FALSE, progress = FALSE)
  })

  expect_s3_class(fit, "prior_var")
})

test_that("prior_belief_var coef_mat has reasonable values", {
  skip_on_cran()

  set.seed(123)
  n <- 100
  p <- 4
  Y <- matrix(rnorm(n * p), n, p)
  colnames(Y) <- paste0("V", 1:p)

  suppressMessages({
    fit <- prior_belief_var(Y, est_ggm = FALSE, progress = FALSE)
  })

  # Coefficients should be finite
  expect_true(all(is.finite(fit$coef_mat)))

  # For random data, coefficients should be relatively small
  expect_true(all(abs(fit$coef_mat) < 1))
})

test_that("prior_belief_var with real ifit data", {
  skip_on_cran()

  # Use built-in ifit data
  y <- na.omit(subset(BGGM::ifit, id == 1)[, 2:5])

  if (nrow(y) > 50) {
    suppressMessages({
      fit <- prior_belief_var(y, est_ggm = FALSE, progress = FALSE)
    })

    expect_s3_class(fit, "prior_var")
    expect_equal(dim(fit$adj), c(ncol(y), ncol(y)))
  }
})
