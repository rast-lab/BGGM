library(testthat)
library(BGGM)

# ============================================
# Tests for fisher_r_to_z
# ============================================

test_that("fisher_r_to_z works with single value", {
  result <- fisher_r_to_z(0.5)

  expect_true(is.numeric(result))
  expect_length(result, 1)
})
test_that("fisher_r_to_z returns correct transformation", {
  r <- 0.5
  z <- fisher_r_to_z(r)

  # Fisher's z = 0.5 * log((1 + r) / (1 - r))
  expected <- 0.5 * log((1 + r) / (1 - r))

  expect_equal(z, expected, tolerance = 1e-10)
})

test_that("fisher_r_to_z works with vector input", {
  r_values <- c(0.1, 0.3, 0.5, 0.7, 0.9)
  result <- fisher_r_to_z(r_values)

  expect_true(is.numeric(result))
  expect_length(result, 5)
})

test_that("fisher_r_to_z handles zero correlation", {
  result <- fisher_r_to_z(0)

  expect_equal(result, 0, tolerance = 1e-10)
})

test_that("fisher_r_to_z handles negative correlations", {
  result_pos <- fisher_r_to_z(0.5)
  result_neg <- fisher_r_to_z(-0.5)

  expect_equal(result_neg, -result_pos, tolerance = 1e-10)
})

test_that("fisher_r_to_z handles extreme values", {
  # Close to 1
  result_high <- fisher_r_to_z(0.99)
  expect_true(is.finite(result_high))
  expect_true(result_high > 0)

  # Close to -1
  result_low <- fisher_r_to_z(-0.99)
  expect_true(is.finite(result_low))
  expect_true(result_low < 0)
})

test_that("fisher_r_to_z with matrix input", {
  r_matrix <- matrix(c(1, 0.5, 0.3, 0.5, 1, 0.4, 0.3, 0.4, 1), 3, 3)
  result <- fisher_r_to_z(r_matrix)

  expect_true(is.matrix(result))
  expect_equal(dim(result), c(3, 3))
})

# ============================================
# Tests for fisher_z_to_r
# ============================================

test_that("fisher_z_to_r works with single value", {
  result <- fisher_z_to_r(0.5)

  expect_true(is.numeric(result))
  expect_length(result, 1)
})

test_that("fisher_z_to_r returns correct transformation", {
  z <- 0.5
  r <- fisher_z_to_r(z)

  # r = (exp(2z) - 1) / (exp(2z) + 1) = tanh(z)
  expected <- tanh(z)

  expect_equal(r, expected, tolerance = 1e-10)
})

test_that("fisher_z_to_r works with vector input", {
  z_values <- c(0.1, 0.3, 0.5, 0.8, 1.5)
  result <- fisher_z_to_r(z_values)

  expect_true(is.numeric(result))
  expect_length(result, 5)
})

test_that("fisher_z_to_r handles zero", {
  result <- fisher_z_to_r(0)

  expect_equal(result, 0, tolerance = 1e-10)
})

test_that("fisher_z_to_r handles negative z values", {
  result_pos <- fisher_z_to_r(0.5)
  result_neg <- fisher_z_to_r(-0.5)

  expect_equal(result_neg, -result_pos, tolerance = 1e-10)
})

test_that("fisher_z_to_r returns values in [-1, 1]", {
  z_values <- c(-3, -1, 0, 1, 3)
  result <- fisher_z_to_r(z_values)

  expect_true(all(result >= -1 & result <= 1))
})

test_that("fisher_z_to_r with matrix input", {
  z_matrix <- matrix(c(Inf, 0.5, 0.3, 0.5, Inf, 0.4, 0.3, 0.4, Inf), 3, 3)
  # Note: Inf will give 1 (tanh(Inf) = 1)
  result <- fisher_z_to_r(z_matrix)

  expect_true(is.matrix(result))
  expect_equal(dim(result), c(3, 3))
})

# ============================================
# Tests for inverse relationship
# ============================================

test_that("fisher_z_to_r inverts fisher_r_to_z", {
  r_original <- c(0.1, 0.3, 0.5, 0.7, -0.2, -0.5)

  z_transformed <- fisher_r_to_z(r_original)
  r_back <- fisher_z_to_r(z_transformed)

  expect_equal(r_back, r_original, tolerance = 1e-10)
})

test_that("fisher_r_to_z inverts fisher_z_to_r", {
  z_original <- c(0.1, 0.5, 1.0, 1.5, -0.3, -1.0)

  r_transformed <- fisher_z_to_r(z_original)
  z_back <- fisher_r_to_z(r_transformed)

  expect_equal(z_back, z_original, tolerance = 1e-10)
})

# ============================================
# Tests for edge cases
# ============================================

test_that("fisher_r_to_z handles NA values", {
  result <- fisher_r_to_z(c(0.5, NA, 0.3))

  expect_true(is.na(result[2]))
  expect_true(!is.na(result[1]))
  expect_true(!is.na(result[3]))
})

test_that("fisher_z_to_r handles NA values", {
  result <- fisher_z_to_r(c(0.5, NA, 0.3))

  expect_true(is.na(result[2]))
  expect_true(!is.na(result[1]))
  expect_true(!is.na(result[3]))
})

# ============================================
# Tests for use with BGGM objects
# ============================================

test_that("fisher_r_to_z works with pcor_mat from estimate", {
  set.seed(123)
  test_data <- matrix(rnorm(50 * 4), ncol = 4)
  colnames(test_data) <- paste0("V", 1:4)

  fit <- estimate(test_data, iter = 50, progress = FALSE)
  pcor <- pcor_mat(fit)

  # Transform partial correlations
  z_pcor <- fisher_r_to_z(pcor)

  expect_true(is.matrix(z_pcor))
  expect_equal(dim(z_pcor), dim(pcor))
})

test_that("fisher transformations preserve matrix structure", {
  set.seed(456)
  test_data <- matrix(rnorm(50 * 4), ncol = 4)
  colnames(test_data) <- paste0("V", 1:4)

  fit <- estimate(test_data, iter = 50, progress = FALSE)
  pcor <- pcor_mat(fit)

  z_pcor <- fisher_r_to_z(pcor)
  pcor_back <- fisher_z_to_r(z_pcor)

  # Should recover original partial correlations
  expect_equal(pcor_back, pcor, tolerance = 1e-10)
})

# ============================================
# Numerical stability tests
# ============================================

test_that("fisher_r_to_z is numerically stable near boundaries", {
  # Very close to 1 but not exactly 1
  r_near_1 <- 0.9999999

  result <- fisher_r_to_z(r_near_1)

  expect_true(is.finite(result))
  expect_true(result > 0)
})

test_that("fisher_z_to_r is numerically stable for large z", {
  # Large positive z should give value close to 1
  z_large <- 10

  result <- fisher_z_to_r(z_large)

  expect_true(is.finite(result))
  expect_true(result > 0.99)
  expect_true(result <= 1)
})
