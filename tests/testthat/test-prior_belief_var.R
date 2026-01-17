library(testthat)
library(BGGM)

# ============================================
# Tests for prior_belief_var()
# ============================================

test_that("prior_belief_var returns correct class", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(rnorm(n * p), n, p)

  suppressMessages({
    fit <- prior_belief_var(Y, est_ggm = FALSE, progress = FALSE)
  })

  expect_s3_class(fit, "BGGM")
  expect_s3_class(fit, "prior_var")
})

test_that("prior_belief_var without GGM returns expected components", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(rnorm(n * p), n, p)

  suppressMessages({
    fit <- prior_belief_var(Y, est_ggm = FALSE, progress = FALSE)
  })

  expect_true("adj" %in% names(fit))
  expect_true("post_prob" %in% names(fit))
  expect_true("coef_mat" %in% names(fit))
})

test_that("prior_belief_var with GGM returns expected components", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(rnorm(n * p), n, p)

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
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(rnorm(n * p), n, p)

  suppressMessages({
    fit <- prior_belief_var(Y, est_ggm = FALSE, progress = FALSE)
  })

  expect_equal(dim(fit$adj), c(p, p))
  expect_equal(dim(fit$post_prob), c(p, p))
  expect_equal(dim(fit$coef_mat), c(p, p))
})

test_that("prior_belief_var post_prob is in [0,1]", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(rnorm(n * p), n, p)

  suppressMessages({
    fit <- prior_belief_var(Y, est_ggm = FALSE, progress = FALSE)
  })

  expect_true(all(fit$post_prob >= 0 & fit$post_prob <= 1))
})

test_that("prior_belief_var adjacency is binary", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(rnorm(n * p), n, p)

  suppressMessages({
    fit <- prior_belief_var(Y, est_ggm = FALSE, progress = FALSE)
  })

  expect_true(all(fit$adj %in% c(0, 1)))
})

test_that("prior_belief_var respects post_odds_cut", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(rnorm(n * p), n, p)

  suppressMessages({
    fit_low <- prior_belief_var(Y, post_odds_cut = 1, est_ggm = FALSE, progress = FALSE)
    fit_high <- prior_belief_var(Y, post_odds_cut = 10, est_ggm = FALSE, progress = FALSE)
  })

  # High cutoff should have fewer or equal edges
  expect_true(sum(fit_high$adj) <= sum(fit_low$adj))
})

test_that("prior_belief_var with custom prior_temporal", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(rnorm(n * p), n, p)

  prior_var <- matrix(1, p, p)
  prior_var[1, 2] <- 10

  suppressMessages({
    fit <- prior_belief_var(Y, prior_temporal = prior_var, est_ggm = FALSE, progress = FALSE)
  })

  expect_s3_class(fit, "prior_var")
})

test_that("prior_belief_var errors with zero in prior_temporal", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(rnorm(n * p), n, p)

  prior_var <- matrix(1, p, p)
  prior_var[1, 2] <- 0

  expect_error(
    suppressMessages(prior_belief_var(Y, prior_temporal = prior_var, progress = FALSE)),
    "zeros are not allowed"
  )
})

test_that("prior_belief_var errors with zero in prior_ggm", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(rnorm(n * p), n, p)

  prior_ggm <- matrix(1, p, p)
  prior_ggm[1, 2] <- 0

  expect_error(
    suppressMessages(prior_belief_var(Y, prior_ggm = prior_ggm, est_ggm = TRUE, progress = FALSE)),
    "zeros are not allowed"
  )
})

test_that("prior_belief_var print method works", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(rnorm(n * p), n, p)

  suppressMessages({
    fit <- prior_belief_var(Y, est_ggm = FALSE, progress = FALSE)
  })

  output <- capture.output(print(fit))

  expect_true(length(output) > 0)
  expect_true(any(grepl("BGGM", output)))
})

test_that("prior_belief_var handles data frame input", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- data.frame(matrix(rnorm(n * p), n, p))

  suppressMessages({
    fit <- prior_belief_var(Y, est_ggm = FALSE, progress = FALSE)
  })

  expect_s3_class(fit, "prior_var")
})

test_that("prior_belief_var coef_mat has reasonable values", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(rnorm(n * p), n, p)

  suppressMessages({
    fit <- prior_belief_var(Y, est_ggm = FALSE, progress = FALSE)
  })

  expect_true(all(is.finite(fit$coef_mat)))
})

test_that("prior_belief_var with custom prior_ggm", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(rnorm(n * p), n, p)

  prior_ggm <- matrix(1, p, p)
  prior_ggm[1, 2] <- 5
  prior_ggm[2, 1] <- 5

  suppressMessages({
    fit <- prior_belief_var(Y, prior_ggm = prior_ggm, est_ggm = TRUE, progress = FALSE)
  })

  expect_s3_class(fit, "prior_var")
  expect_true("adj_ggm" %in% names(fit))
})

test_that("prior_belief_var temporal adjacency dimensions match", {
  set.seed(123)
  n <- 50
  p <- 4
  Y <- matrix(rnorm(n * p), n, p)

  suppressMessages({
    fit <- prior_belief_var(Y, est_ggm = TRUE, progress = FALSE)
  })

  expect_equal(dim(fit$adj_temporal), c(p, p))
  expect_equal(dim(fit$post_prob_temporal), c(p, p))
})

test_that("prior_belief_var GGM adjacency is symmetric", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(rnorm(n * p), n, p)

  suppressMessages({
    fit <- prior_belief_var(Y, est_ggm = TRUE, progress = FALSE)
  })

  # GGM adjacency should be symmetric
expect_true(isSymmetric(fit$adj_ggm))
})

test_that("prior_belief_var with progress = TRUE works", {
  set.seed(123)
  n <- 30
  p <- 3
  Y <- matrix(rnorm(n * p), n, p)

  expect_no_error({
    capture.output({
      suppressMessages({
        fit <- prior_belief_var(Y, est_ggm = FALSE, progress = TRUE)
      })
    })
  })
})
