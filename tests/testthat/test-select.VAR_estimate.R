library(testthat)
library(BGGM)

# ============================================
# Tests for select.var_estimate()
# ============================================

test_that("select.var_estimate returns correct class", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  expect_s3_class(sel, "BGGM")
  expect_s3_class(sel, "select.var_estimate")
})

test_that("select.var_estimate returns expected components", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  expect_true("pcor_adj" %in% names(sel))
  expect_true("beta_adj" %in% names(sel))
  expect_true("pcor_weighted_adj" %in% names(sel))
  expect_true("beta_weighted_adj" %in% names(sel))
  expect_true("pcor_mu" %in% names(sel))
  expect_true("beta_mu" %in% names(sel))
})

test_that("select.var_estimate pcor_adj is binary", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  expect_true(all(sel$pcor_adj %in% c(0, 1)))
})

test_that("select.var_estimate beta_adj is binary", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  expect_true(all(sel$beta_adj %in% c(0, 1)))
})

test_that("select.var_estimate pcor_adj has correct dimensions", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]
  p <- ncol(Y)

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  expect_equal(dim(sel$pcor_adj), c(p, p))
})

test_that("select.var_estimate respects cred parameter", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)

  sel_95 <- select(fit, cred = 0.95)
  sel_99 <- select(fit, cred = 0.99)

  # Higher cred should have fewer or equal selected edges
  expect_true(sum(sel_99$pcor_adj) <= sum(sel_95$pcor_adj))
})

test_that("select.var_estimate alternative = greater works", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "greater")

  expect_s3_class(sel, "select.var_estimate")
  expect_equal(sel$alternative, "greater")
})

test_that("select.var_estimate alternative = less works", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "less")

  expect_s3_class(sel, "select.var_estimate")
  expect_equal(sel$alternative, "less")
})

test_that("select.var_estimate pcor_weighted_adj matches selection", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  # Where pcor_adj is 0, pcor_weighted_adj should also be 0
  zero_positions <- which(sel$pcor_adj == 0)
  expect_true(all(sel$pcor_weighted_adj[zero_positions] == 0))
})

test_that("select.var_estimate beta_weighted_adj matches selection", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  # Where beta_adj is 0, beta_weighted_adj should also be 0
  zero_positions <- which(sel$beta_adj == 0)
  expect_true(all(sel$beta_weighted_adj[zero_positions] == 0))
})

test_that("select.var_estimate stores cred value", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit, cred = 0.90)

  expect_equal(sel$cred, 0.90)
})

test_that("select.var_estimate print method works", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  output <- capture.output(print(sel))

  expect_true(length(output) > 0)
  expect_true(any(grepl("BGGM", output)))
  expect_true(any(grepl("VAR", output)))
})

test_that("select.var_estimate print shows partial correlations", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  output <- capture.output(print(sel))

  expect_true(any(grepl("Partial Correlations", output)))
})

test_that("select.var_estimate print shows coefficients", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  output <- capture.output(print(sel))

  expect_true(any(grepl("Coefficients", output)))
})

test_that("select.var_estimate pcor_mu is symmetric", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  expect_true(isSymmetric(sel$pcor_mu))
})

test_that("select.var_estimate pcor_mu diagonal is approximately zero", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  # Diagonal should be very small (close to zero)
  expect_true(all(abs(diag(sel$pcor_mu)) < 0.1))
})

test_that("select.var_estimate alternative greater selects positive", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "greater")

  # Selected edges should have positive posterior mean
  selected <- which(sel$pcor_adj == 1, arr.ind = TRUE)

  if (nrow(selected) > 0) {
    for (i in 1:nrow(selected)) {
      r <- selected[i, 1]
      c <- selected[i, 2]
      expect_true(sel$pcor_mu[r, c] > 0)
    }
  }
})

test_that("select.var_estimate alternative less selects negative", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "less")

  # Selected edges should have negative posterior mean
  selected <- which(sel$pcor_adj == 1, arr.ind = TRUE)

  if (nrow(selected) > 0) {
    for (i in 1:nrow(selected)) {
      r <- selected[i, 1]
      c <- selected[i, 2]
      expect_true(sel$pcor_mu[r, c] < 0)
    }
  }
})

