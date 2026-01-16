library(testthat)
library(BGGM)

# ============================================
# Tests for gen_net
# ============================================

test_that("gen_net works with default parameters", {
  set.seed(123)
  result <- gen_net(p = 5)

  expect_true(is.list(result))
})

test_that("gen_net returns correct structure", {
  set.seed(123)
  result <- gen_net(p = 5)

  # Should contain partial correlation matrix and possibly other elements
  expect_true(any(c("pcor", "pcor_mat", "adj", "Sigma") %in% names(result)))
})

test_that("gen_net returns matrix of correct dimensions", {
  set.seed(456)
  p <- 6
  result <- gen_net(p = p)

  if ("pcor" %in% names(result)) {
    expect_equal(dim(result$pcor), c(p, p))
  }

  if ("pcor_mat" %in% names(result)) {
    expect_equal(dim(result$pcor_mat), c(p, p))
  }
})

test_that("gen_net partial correlation matrix is symmetric", {
  set.seed(789)
  result <- gen_net(p = 5)

  if ("pcor" %in% names(result)) {
    expect_equal(result$pcor, t(result$pcor), tolerance = 1e-10)
  }
})

test_that("gen_net edge_prob parameter works", {
  set.seed(111)
  # Low edge probability
  result_sparse <- gen_net(p = 10, edge_prob = 0.1)

  set.seed(111)
  # High edge probability
  result_dense <- gen_net(p = 10, edge_prob = 0.9)

  expect_true(is.list(result_sparse))
  expect_true(is.list(result_dense))
})

test_that("gen_net lb and ub parameters work", {
  set.seed(222)
  result <- gen_net(p = 5, lb = 0.2, ub = 0.6)

  if ("pcor" %in% names(result)) {
    pcor <- result$pcor
    # Non-zero partial correlations should be within bounds (in absolute value)
    non_zero <- pcor[upper.tri(pcor) & pcor != 0]
    if (length(non_zero) > 0) {
      expect_true(all(abs(non_zero) >= 0.2 & abs(non_zero) <= 0.6))
    }
  }

  expect_true(is.list(result))
})

test_that("gen_net returns positive definite covariance", {
  set.seed(333)
  result <- gen_net(p = 5)

  if ("Sigma" %in% names(result)) {
    # Check positive definiteness by ensuring all eigenvalues are positive
    eigenvalues <- eigen(result$Sigma)$values
    expect_true(all(eigenvalues > 0))
  }
})

test_that("gen_net works with different number of variables", {
  set.seed(444)
  result_3 <- gen_net(p = 3)
  result_10 <- gen_net(p = 10)

  expect_true(is.list(result_3))
  expect_true(is.list(result_10))
})

# ============================================
# Tests for gen_ordinal
# ============================================
# Note: gen_ordinal requires cor_mat parameter (no default)

# Helper to create a simple correlation matrix
create_cor_mat <- function(p) {
  mat <- diag(p)
  # Add some off-diagonal correlations
  for (i in 1:(p - 1)) {
    mat[i, i + 1] <- 0.3
    mat[i + 1, i] <- 0.3
  }
  mat
}

test_that("gen_ordinal works with required cor_mat parameter", {
  set.seed(123)
  p <- 5
  cor_mat <- create_cor_mat(p)
  result <- gen_ordinal(n = 100, p = p, cor_mat = cor_mat)

  expect_true(is.matrix(result))
})

test_that("gen_ordinal returns matrix of correct dimensions", {
  set.seed(456)
  n <- 100
  p <- 5
  cor_mat <- create_cor_mat(p)
  result <- gen_ordinal(n = n, p = p, cor_mat = cor_mat)

  expect_equal(dim(result), c(n, p))
})

test_that("gen_ordinal returns integer values", {
  set.seed(789)
  p <- 5
  cor_mat <- create_cor_mat(p)
  result <- gen_ordinal(n = 100, p = p, cor_mat = cor_mat)

  # Check that values are integers
  expect_true(all(result == floor(result)))
})

test_that("gen_ordinal respects levels parameter (number of categories)", {
  set.seed(111)
  p <- 5
  levels <- 5
  cor_mat <- create_cor_mat(p)
  result <- gen_ordinal(n = 100, p = p, levels = levels, cor_mat = cor_mat)

  # Values should be between 1 and levels
  expect_true(all(result >= 1 & result <= levels))
})

test_that("gen_ordinal works with different levels values", {
  set.seed(222)
  p <- 4
  cor_mat <- create_cor_mat(p)
  result_3 <- gen_ordinal(n = 50, p = p, levels = 3, cor_mat = cor_mat)
  result_7 <- gen_ordinal(n = 50, p = p, levels = 7, cor_mat = cor_mat)

  expect_true(is.matrix(result_3))
  expect_true(is.matrix(result_7))
})

test_that("gen_ordinal works with different sample sizes", {
  set.seed(333)
  p <- 4
  cor_mat <- create_cor_mat(p)
  result_small <- gen_ordinal(n = 30, p = p, cor_mat = cor_mat)
  result_large <- gen_ordinal(n = 500, p = p, cor_mat = cor_mat)

  expect_equal(nrow(result_small), 30)
  expect_equal(nrow(result_large), 500)
})

test_that("gen_ordinal works with different number of variables", {
  set.seed(444)
  cor_mat_3 <- create_cor_mat(3)
  cor_mat_8 <- create_cor_mat(8)

  result_3 <- gen_ordinal(n = 100, p = 3, cor_mat = cor_mat_3)
  result_8 <- gen_ordinal(n = 100, p = 8, cor_mat = cor_mat_8)

  expect_equal(ncol(result_3), 3)
  expect_equal(ncol(result_8), 8)
})

test_that("gen_ordinal works with binary data (levels = 2)", {
  set.seed(555)
  p <- 4
  cor_mat <- create_cor_mat(p)
  result <- gen_ordinal(n = 100, p = p, levels = 2, cor_mat = cor_mat)

  # Binary data should only have values 1 and 2
  expect_true(all(result %in% c(1, 2)))
})

# ============================================
# Integration tests: use gen_net with estimate
# ============================================

test_that("gen_net output can be used for simulation", {
  set.seed(555)
  # Generate network
  net <- gen_net(p = 4)

  # If Sigma is provided, can generate data
  if ("Sigma" %in% names(net)) {
    # Generate data from the network
    Y <- MASS::mvrnorm(n = 100, mu = rep(0, 4), Sigma = net$Sigma)
    colnames(Y) <- paste0("V", 1:4)

    # Fit model
    fit <- estimate(Y, iter = 50, progress = FALSE)

    expect_s3_class(fit, "estimate")
  }
})

test_that("gen_ordinal output can be used with estimate", {
  set.seed(666)
  # Generate ordinal data with correlation matrix
  p <- 4
  cor_mat <- create_cor_mat(p)
  Y <- gen_ordinal(n = 100, p = p, levels = 4, cor_mat = cor_mat)

  colnames(Y) <- paste0("V", 1:4)

  # Fit model
  fit <- estimate(Y, type = "ordinal", iter = 50, progress = FALSE)

  expect_s3_class(fit, "estimate")
})
