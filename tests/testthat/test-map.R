library(testthat)
library(BGGM)

# ============================================
# Tests for map() - Maximum A Posteriori
# ============================================

test_that("map returns correct class", {
  set.seed(123)
  Y <- matrix(rnorm(100), 20, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- map(Y)

  expect_s3_class(fit, "BGGM")
  expect_s3_class(fit, "map")
  expect_s3_class(fit, "estimate")
})

test_that("map returns expected components", {
  set.seed(123)
  Y <- matrix(rnorm(100), 20, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- map(Y)

  expect_true("precision" %in% names(fit))
  expect_true("pcor" %in% names(fit))
  expect_true("betas" %in% names(fit))
  expect_true("dat" %in% names(fit))
})

test_that("map precision matrix has correct dimensions", {
  set.seed(123)
  p <- 5
  Y <- matrix(rnorm(100), 20, p)
  colnames(Y) <- paste0("V", 1:p)

  fit <- map(Y)

  expect_equal(dim(fit$precision), c(p, p))
  expect_equal(dim(fit$pcor), c(p, p))
})

test_that("map pcor matrix is symmetric", {
  set.seed(123)
  Y <- matrix(rnorm(100), 20, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- map(Y)

  expect_true(isSymmetric(fit$pcor))
})

test_that("map handles data frames", {
  set.seed(123)
  Y <- data.frame(matrix(rnorm(100), 20, 5))
  colnames(Y) <- paste0("V", 1:5)

  fit <- map(Y)

  expect_s3_class(fit, "map")
})

test_that("map removes NA values", {
  set.seed(123)
  Y <- matrix(rnorm(100), 20, 5)
  colnames(Y) <- paste0("V", 1:5)
  Y[1, 1] <- NA
  Y[2, 3] <- NA

  fit <- map(Y)

  # Should have processed without error

  expect_s3_class(fit, "map")
  # Data should have fewer rows
  expect_equal(nrow(fit$dat), 18)
})

test_that("map betas have correct dimensions", {
  set.seed(123)
  p <- 5
  Y <- matrix(rnorm(100), 20, p)
  colnames(Y) <- paste0("V", 1:p)

  fit <- map(Y)

  # betas: p rows (one per node), p-1 columns (predictors)
  expect_equal(nrow(fit$betas), p)
  expect_equal(ncol(fit$betas), p - 1)
})

test_that("map precision matrix is positive definite", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- map(Y)

  # Check positive definiteness via eigenvalues
  eigenvalues <- eigen(fit$precision, only.values = TRUE)$values
  expect_true(all(eigenvalues > 0))
})

test_that("map works with bfi data", {
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- map(Y)

  expect_s3_class(fit, "map")
  expect_equal(dim(fit$pcor), c(5, 5))
})

test_that("map print method works", {
  set.seed(123)
  Y <- matrix(rnorm(100), 20, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- map(Y)

  # Capture print output
  output <- capture.output(print(fit))

  expect_true(length(output) > 0)
  expect_true(any(grepl("BGGM", output)))
})

test_that("map pcor values are in valid range", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- map(Y)

  # Off-diagonal should be in [-1, 1]
  off_diag <- fit$pcor[upper.tri(fit$pcor)]
  expect_true(all(off_diag >= -1 & off_diag <= 1))
})
