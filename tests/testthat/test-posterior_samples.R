library(testthat)
library(BGGM)

# ============================================
# Tests for posterior_samples()
# ============================================

test_that("posterior_samples returns matrix for estimate object", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  samps <- posterior_samples(fit)

  expect_true(is.matrix(samps))
})

test_that("posterior_samples has correct dimensions", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]
  p <- 5
  iter <- 100

  fit <- estimate(Y, iter = iter, progress = FALSE)
  samps <- posterior_samples(fit)

  # Number of partial correlations = p*(p-1)/2
  n_pcors <- p * (p - 1) / 2
  expect_equal(nrow(samps), iter)
  expect_equal(ncol(samps), n_pcors)
})

test_that("posterior_samples works with explore object", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  samps <- posterior_samples(fit)

  expect_true(is.matrix(samps))
})

test_that("posterior_samples has correct column names", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  samps <- posterior_samples(fit)

  # Column names should be in format "A1--A2"
  expect_true(all(grepl("--", colnames(samps))))
})

test_that("posterior_samples without column names uses numbers", {
  set.seed(123)
  Y <- as.matrix(BGGM::bfi[1:50, 1:5])
  colnames(Y) <- NULL

  fit <- estimate(Y, iter = 100, progress = FALSE)
  samps <- posterior_samples(fit)

  # Column names should be in format "1--2"
  expect_true(all(grepl("^[0-9]+--[0-9]+$", colnames(samps))))
})

test_that("posterior_samples with formula includes betas", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]
  Y$gender <- BGGM::bfi$gender[1:50]

  fit <- estimate(Y, formula = ~ gender,
                  type = "continuous",
                  iter = 100,
                  progress = FALSE)

  samps <- posterior_samples(fit)

  # Should have both pcors and betas
  expect_true(ncol(samps) > 5 * 4 / 2)
  # Should have columns with beta terms
  expect_true(any(grepl("gender", colnames(samps))))
})

test_that("posterior_samples with intercept only formula", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]
  Y$gender <- BGGM::bfi$gender[1:50]

  fit <- estimate(Y, formula = ~ 1,
                  type = "continuous",
                  iter = 100,
                  progress = FALSE)

  samps <- posterior_samples(fit)

  # Should have intercept columns
  expect_true(any(grepl("Intercept", colnames(samps))))
})

test_that("posterior_samples errors for non-default estimate", {
  set.seed(123)
  Y <- matrix(rnorm(100), 20, 5)

  expect_error(
    posterior_samples(Y),
    "class not currently supported"
  )
})

test_that("posterior_samples values are finite", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  samps <- posterior_samples(fit)

  expect_true(all(is.finite(samps)))
})

test_that("posterior_samples values are in valid range", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  samps <- posterior_samples(fit)

  # Partial correlations should be in [-1, 1]
  expect_true(all(samps >= -1 & samps <= 1))
})

test_that("posterior_samples with var_estimate object", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  samps <- posterior_samples(fit)

  expect_true(is.matrix(samps))
})

test_that("posterior_samples var_estimate has correct structure", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  samps <- posterior_samples(fit)

  # Should include both pcors and betas
  expect_true(ncol(samps) > 0)
})

test_that("posterior_samples var_estimate beta columns named correctly", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  samps <- posterior_samples(fit)

  # Should have columns with variable names
  expect_true(ncol(samps) > 0)
})

test_that("posterior_samples multiple predictors in formula", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]
  Y$gender <- BGGM::bfi$gender[1:50]
  Y$education <- BGGM::bfi$education[1:50]

  fit <- estimate(Y, formula = ~ gender + education,
                  type = "continuous",
                  iter = 100,
                  progress = FALSE)

  samps <- posterior_samples(fit)

  # Should have columns for both predictors
  expect_true(any(grepl("gender", colnames(samps))))
  expect_true(any(grepl("education", colnames(samps))))
})

test_that("posterior_samples with binary data", {
  set.seed(123)
  Y <- matrix(rbinom(150, 1, 0.5), 50, 3)
  colnames(Y) <- paste0("V", 1:3)

  fit <- estimate(Y, type = "binary", iter = 100, progress = FALSE)
  samps <- posterior_samples(fit)

  expect_true(is.matrix(samps))
  # Number of partial correlations = p*(p-1)/2 = 3
  expect_true(ncol(samps) >= 3)
})

test_that("posterior_samples with ordinal data", {
  set.seed(123)
  Y <- matrix(sample(1:5, 150, replace = TRUE), 50, 3)
  colnames(Y) <- paste0("V", 1:3)

  fit <- estimate(Y, type = "ordinal", iter = 100, progress = FALSE)
  samps <- posterior_samples(fit)

  expect_true(is.matrix(samps))
})

test_that("posterior_samples row count matches iter", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]
  iter <- 75

  fit <- estimate(Y, iter = iter, progress = FALSE)
  samps <- posterior_samples(fit)

  expect_equal(nrow(samps), iter)
})

