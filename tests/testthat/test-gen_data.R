library(testthat)
library(BGGM)

# ============================================
# Tests for gen_net
# ============================================

test_that("gen_net works with default parameters", {
  set.seed(123)
  result <- gen_net(p = 5)

  expect_true(is.list(result))
  expect_true(all(c("pcors", "cors", "adj") %in% names(result)))
})

test_that("gen_net returns correct structure", {
  set.seed(123)
  result <- gen_net(p = 5)

  expect_true(all(c("pcors", "cors", "adj") %in% names(result)))
})

test_that("gen_net returns matrix of correct dimensions", {
  set.seed(456)
  p <- 6
  result <- gen_net(p = p)

  expect_equal(dim(result$pcors), c(p, p))
  expect_equal(dim(result$adj),   c(p, p))
  expect_equal(dim(result$cors),  c(p, p))
})

test_that("gen_net partial correlation matrix is symmetric", {
  set.seed(789)
  result <- gen_net(p = 5)

  expect_equal(result$pcors, t(result$pcors), tolerance = 1e-10)
})

test_that("gen_net edge_prob parameter affects density", {
  set.seed(111)
  result_sparse <- gen_net(p = 10, edge_prob = 0.1)
  set.seed(222)
  result_dense  <- gen_net(p = 10, edge_prob = 0.9)

  density_sparse <- mean(result_sparse$adj[upper.tri(result_sparse$adj)])
  density_dense  <- mean(result_dense$adj[upper.tri(result_dense$adj)])
  expect_true(density_sparse < density_dense)
})

test_that("gen_net lb and ub parameters constrain partial correlations", {
  set.seed(222)
  result <- gen_net(p = 5, lb = 0.2, ub = 0.6)

  non_zero <- result$pcors[upper.tri(result$pcors) & result$pcors != 0]
  if (length(non_zero) > 0) {
    expect_true(all(abs(non_zero) >= 0.2 & abs(non_zero) <= 0.6))
  }
})

test_that("gen_net correlation matrix is positive definite", {
  set.seed(333)
  result <- gen_net(p = 5)

  eigenvalues <- eigen(result$cors)$values
  expect_true(all(eigenvalues > 0))
})

test_that("gen_net works with different number of variables", {
  set.seed(444)
  result_3  <- gen_net(p = 3)
  result_10 <- gen_net(p = 10)

  expect_equal(dim(result_3$pcors),  c(3,  3))
  expect_equal(dim(result_10$pcors), c(10, 10))
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
  net <- gen_net(p = 4)

  Y <- MASS::mvrnorm(n = 100, mu = rep(0, 4), Sigma = net$cors)
  colnames(Y) <- paste0("V", 1:4)

  fit <- estimate(Y, iter = 50, progress = FALSE)
  expect_s3_class(fit, "estimate")
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
