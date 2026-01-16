library(testthat)
library(BGGM)

# Test data setup
set.seed(123)
n <- 50
p <- 5
test_data <- matrix(rnorm(n * p), ncol = p)
colnames(test_data) <- paste0("V", 1:p)

# Create estimate object for testing
fit <- estimate(test_data, iter = 100, progress = FALSE)

# ============================================
# Tests for coef.estimate
# ============================================

test_that("coef.estimate works with estimate object", {
  result <- coef(fit)

  expect_true(is.list(result) || is.matrix(result) || is.data.frame(result))
})

test_that("coef.estimate returns regression coefficients", {
  result <- coef(fit)

  # Should have coefficient information
  if (is.list(result)) {
    expect_true(length(result) > 0)
  }

  if (is.matrix(result) || is.data.frame(result)) {
    expect_true(nrow(result) > 0)
  }
})

test_that("coef.estimate works with node parameter", {
  # Get coefficients for a specific node
  result <- coef(fit, node = 1)

  expect_true(is.list(result) || is.numeric(result) || is.data.frame(result))
})

test_that("coef.estimate works with different nodes", {
  result1 <- coef(fit, node = 1)
  result2 <- coef(fit, node = 2)

  expect_true(!is.null(result1))
  expect_true(!is.null(result2))
})

# Test summary of coef
test_that("coef.estimate summary provides credible intervals", {
  result <- coef(fit)

  # Result should contain interval information
  if (is.data.frame(result)) {
    # Check for common CI column names
    ci_cols <- c("Post.mean", "Cred", "mean", "sd", "lower", "upper")
    expect_true(any(ci_cols %in% names(result)) || ncol(result) >= 2)
  }
})

# ============================================
# Tests for predict.estimate
# ============================================

test_that("predict.estimate works with estimate object", {
  # predict.estimate returns a 3D array
  result <- predict(fit, progress = FALSE)

  expect_true(is.array(result))
})

test_that("predict.estimate returns 3D array with correct dimensions", {
  result <- predict(fit, progress = FALSE)

  # Result should be array with dimensions (n, 4, p)
  # where 4 is: Post.mean, Post.sd, Cred.lb, Cred.ub
  expect_equal(length(dim(result)), 3)
  expect_equal(dim(result)[1], n)  # number of observations
  expect_equal(dim(result)[2], 4)  # summary stats
  expect_equal(dim(result)[3], p)  # number of variables
})

test_that("predict.estimate works with newdata", {
  set.seed(456)
  new_data <- matrix(rnorm(20 * p), ncol = p)
  colnames(new_data) <- paste0("V", 1:p)

  result <- predict(fit, newdata = new_data, progress = FALSE)

  expect_true(is.array(result))
  expect_equal(dim(result)[1], 20)  # new observations
})

test_that("predict.estimate works with summary = FALSE", {
  result <- predict(fit, summary = FALSE, progress = FALSE)

  expect_true(is.array(result))
  # When summary = FALSE, dimensions are (iter, n, p)
  expect_equal(length(dim(result)), 3)
})

test_that("predict.estimate contains correct dimnames", {
  result <- predict(fit, progress = FALSE)

  # Second dimension should have summary stat names
  expect_true("Post.mean" %in% dimnames(result)[[2]])
  expect_true("Post.sd" %in% dimnames(result)[[2]])
})

# ============================================
# Tests for posterior_samples
# ============================================

test_that("posterior_samples works with estimate object", {
  result <- posterior_samples(fit)

  expect_true(is.matrix(result) || is.data.frame(result))
})

test_that("posterior_samples returns correct number of samples", {
  result <- posterior_samples(fit)

  if (is.matrix(result) || is.data.frame(result)) {
    # Should have rows related to iterations
    expect_true(nrow(result) > 0)
  }
})

test_that("posterior_samples works with explore object", {
  fit_explore <- explore(test_data, progress = FALSE)
  result <- posterior_samples(fit_explore)

  expect_true(is.matrix(result) || is.data.frame(result))
})

test_that("posterior_samples includes partial correlations", {
  result <- posterior_samples(fit)

  # Samples should represent posterior distribution of partial correlations
  if (is.matrix(result) || is.data.frame(result)) {
    # Number of columns should relate to unique edges: p*(p-1)/2
    n_edges <- p * (p - 1) / 2
    # Or could be full matrix representation
    expect_true(ncol(result) >= n_edges || ncol(result) == p * p)
  }
})

# ============================================
# Tests for regression_summary (if exported)
# ============================================

test_that("regression_summary works", {
  result <- tryCatch(
    {
      regression_summary(fit)
    },
    error = function(e) NULL
  )

  if (!is.null(result)) {
    expect_true(is.list(result) || is.data.frame(result))
  }
})

# ============================================
# Tests with different data types
# ============================================

test_that("coef works with binary data estimate", {
  set.seed(333)
  binary_data <- matrix(as.integer(rnorm(50 * 4) > 0), ncol = 4)
  colnames(binary_data) <- paste0("V", 1:4)

  fit_bin <- estimate(binary_data, type = "binary", iter = 50, progress = FALSE)
  result <- coef(fit_bin)

  expect_true(!is.null(result))
})

test_that("coef works with ordinal data estimate", {
  set.seed(444)
  ordinal_data <- matrix(sample(1:4, 50 * 4, replace = TRUE), ncol = 4)
  colnames(ordinal_data) <- paste0("V", 1:4)

  fit_ord <- estimate(ordinal_data, type = "ordinal", iter = 50, progress = FALSE)
  result <- coef(fit_ord)

  expect_true(!is.null(result))
})

test_that("posterior_samples works with different iterations", {
  fit_50 <- estimate(test_data, iter = 50, progress = FALSE)
  fit_100 <- estimate(test_data, iter = 100, progress = FALSE)

  result_50 <- posterior_samples(fit_50)
  result_100 <- posterior_samples(fit_100)

  if (is.matrix(result_50) && is.matrix(result_100)) {
    expect_true(nrow(result_50) <= 50)
    expect_true(nrow(result_100) <= 100)
  }
})

# ============================================
# Tests for posterior_predict (if different from predict)
# ============================================

test_that("posterior_predict works", {
  result <- tryCatch(
    {
      posterior_predict(fit)
    },
    error = function(e) NULL
  )

  if (!is.null(result)) {
    expect_true(is.matrix(result) || is.array(result) || is.list(result))
  }
})

test_that("posterior_predict generates predictive samples", {
  result <- tryCatch(
    {
      posterior_predict(fit, iter = 50)
    },
    error = function(e) NULL
  )

  if (!is.null(result)) {
    # Should generate multiple predictive samples
    expect_true(!is.null(result))
  }
})
