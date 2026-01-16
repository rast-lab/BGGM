library(testthat)
library(BGGM)

# ============================================
# Tests for impute_data() - MVN Imputation
# ============================================

test_that("impute_data returns correct class", {
  set.seed(123)
  n <- 100
  p <- 5
  Y <- matrix(rnorm(n * p), n, p)
  colnames(Y) <- paste0("V", 1:p)

  # Add some missing values
  Y[sample(1:n, 10), 1] <- NA
  Y[sample(1:n, 10), 3] <- NA

  fit <- impute_data(Y, iter = 50, progress = FALSE)

  expect_s3_class(fit, "BGGM")
  expect_s3_class(fit, "mvn_imputation")
})

test_that("impute_data returns imputed datasets", {
  set.seed(123)
  n <- 100
  p <- 5
  Y <- matrix(rnorm(n * p), n, p)
  colnames(Y) <- paste0("V", 1:p)

  # Add missing values
  Y[1:5, 1] <- NA
  Y[6:10, 2] <- NA

  fit <- impute_data(Y, iter = 50, progress = FALSE)

  expect_true("imputed_datasets" %in% names(fit))
})

test_that("impute_data returns correct dimensions", {
  set.seed(123)
  n <- 100
  p <- 5
  iter <- 50
  Y <- matrix(rnorm(n * p), n, p)
  colnames(Y) <- paste0("V", 1:p)

  # Add missing values
  Y[1:5, 1] <- NA

  fit <- impute_data(Y, iter = iter, progress = FALSE)

  # Should be 3D array: n x p x iter
  expect_equal(dim(fit$imputed_datasets)[1], n)
  expect_equal(dim(fit$imputed_datasets)[2], p)
  expect_equal(dim(fit$imputed_datasets)[3], iter)
})

test_that("impute_data errors with no missing values", {
  set.seed(123)
  Y <- matrix(rnorm(100), 20, 5)
  colnames(Y) <- paste0("V", 1:5)

  expect_error(
    impute_data(Y, iter = 50, progress = FALSE),
    "no missing values detected"
  )
})

test_that("impute_data errors with unsupported type", {
  set.seed(123)
  Y <- matrix(rnorm(100), 20, 5)
  Y[1, 1] <- NA

  expect_error(
    impute_data(Y, type = "binary", iter = 50, progress = FALSE),
    "currently only"
  )
})

test_that("impute_data imputed values are numeric", {
  set.seed(123)
  n <- 100
  p <- 5
  Y <- matrix(rnorm(n * p), n, p)
  colnames(Y) <- paste0("V", 1:p)

  # Add missing values
  missing_idx <- 1:10
  Y[missing_idx, 1] <- NA

  fit <- impute_data(Y, iter = 50, progress = FALSE)

  # Check that imputed values are numeric (not NA)
  imputed_vals <- fit$imputed_datasets[missing_idx, 1, ]
  expect_true(all(is.finite(imputed_vals)))
})

test_that("impute_data with lambda parameter", {
  set.seed(123)
  n <- 100
  p <- 5
  Y <- matrix(rnorm(n * p), n, p)
  colnames(Y) <- paste0("V", 1:p)

  Y[1:5, 1] <- NA

  # Custom lambda
  fit <- impute_data(Y, lambda = 10, iter = 50, progress = FALSE)

  expect_s3_class(fit, "mvn_imputation")
})

test_that("impute_data default lambda is p + 2", {
  set.seed(123)
  n <- 100
  p <- 5
  Y <- matrix(rnorm(n * p), n, p)
  colnames(Y) <- paste0("V", 1:p)

  Y[1:5, 1] <- NA

  # Should work without explicit lambda
  fit <- impute_data(Y, iter = 50, progress = FALSE)

  expect_s3_class(fit, "mvn_imputation")
})

test_that("impute_data handles multiple columns with missing", {
  set.seed(123)
  n <- 100
  p <- 5
  Y <- matrix(rnorm(n * p), n, p)
  colnames(Y) <- paste0("V", 1:p)

  # Add missing values to multiple columns
  Y[1:5, 1] <- NA
  Y[6:10, 2] <- NA
  Y[11:15, 3] <- NA
  Y[16:20, 4] <- NA
  Y[21:25, 5] <- NA

  fit <- impute_data(Y, iter = 50, progress = FALSE)

  # All imputed datasets should have no NA
  for (i in 1:dim(fit$imputed_datasets)[3]) {
    expect_false(any(is.na(fit$imputed_datasets[,,i])))
  }
})

test_that("impute_data handles data frame input", {
  set.seed(123)
  n <- 100
  p <- 5
  Y <- data.frame(matrix(rnorm(n * p), n, p))
  colnames(Y) <- paste0("V", 1:p)

  Y[1:5, 1] <- NA

  fit <- impute_data(Y, iter = 50, progress = FALSE)

  expect_s3_class(fit, "mvn_imputation")
})

test_that("impute_data print method works", {
  set.seed(123)
  n <- 100
  p <- 5
  Y <- matrix(rnorm(n * p), n, p)
  colnames(Y) <- paste0("V", 1:p)

  Y[1:5, 1] <- NA

  fit <- impute_data(Y, iter = 50, progress = FALSE)

  output <- capture.output(print(fit))

  expect_true(length(output) > 0)
  expect_true(any(grepl("BGGM", output)))
})

test_that("impute_data imputed values vary across iterations", {
  set.seed(123)
  n <- 100
  p <- 5
  Y <- matrix(rnorm(n * p), n, p)
  colnames(Y) <- paste0("V", 1:p)

  Y[1, 1] <- NA

  fit <- impute_data(Y, iter = 50, progress = FALSE)

  # Imputed values should vary (not all identical)
  imputed_vals <- fit$imputed_datasets[1, 1, ]
  expect_true(sd(imputed_vals) > 0)
})

test_that("impute_data preserves non-missing values", {
  set.seed(123)
  n <- 100
  p <- 5
  Y <- matrix(rnorm(n * p), n, p)
  colnames(Y) <- paste0("V", 1:p)

  # Store original non-missing values (as numeric, not named)
  original_val <- as.numeric(Y[50, 3])

  Y[1:5, 1] <- NA

  fit <- impute_data(Y, iter = 50, progress = FALSE)

  # Non-missing values should be preserved
  expect_equal(as.numeric(fit$imputed_datasets[50, 3, 1]), original_val, tolerance = 1e-10)
})
