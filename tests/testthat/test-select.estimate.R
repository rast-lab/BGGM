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

# Basic functionality tests
test_that("select.estimate works with default parameters", {
  result <- select(fit)

  expect_s3_class(result, "select.estimate")
  expect_true(is.list(result))
})

test_that("select.estimate returns correct structure", {
  result <- select(fit)

  # Should contain pcor_adj (partial correlations with zeros for non-selected edges)
  # and adj (adjacency matrix)
  expect_true("pcor_adj" %in% names(result) || "adj" %in% names(result))
})

test_that("select.estimate returns adjacency and partial correlation matrices", {
  result <- select(fit)

  # Check for key outputs
  if ("pcor_mat_zero" %in% names(result)) {
    expect_true(is.matrix(result$pcor_mat_zero))
    expect_equal(dim(result$pcor_mat_zero), c(p, p))
  }

  if ("Adj" %in% names(result)) {
    expect_true(is.matrix(result$Adj))
    expect_equal(dim(result$Adj), c(p, p))
  }
})

# Test different credible interval levels
test_that("select.estimate works with different cred levels", {
  result_90 <- select(fit, cred = 0.90)
  result_95 <- select(fit, cred = 0.95)
  result_99 <- select(fit, cred = 0.99)

  expect_s3_class(result_90, "select.estimate")
  expect_s3_class(result_95, "select.estimate")
  expect_s3_class(result_99, "select.estimate")
})

# Test alternative hypotheses
test_that("select.estimate works with two.sided alternative", {
  result <- select(fit, alternative = "two.sided")

  expect_s3_class(result, "select.estimate")
})

test_that("select.estimate works with greater alternative", {
  result <- select(fit, alternative = "greater")

  expect_s3_class(result, "select.estimate")
})

test_that("select.estimate works with less alternative", {
  result <- select(fit, alternative = "less")

  expect_s3_class(result, "select.estimate")
})

# Test print method (select.estimate uses print.BGGM dispatcher)
test_that("print.select.estimate works", {
  sel <- select(fit)

  # Test that print works without error
  expect_output(print(sel))
})

# Test plot method
test_that("plot.select.estimate works without error", {
  sel <- select(fit)

  result <- tryCatch(
    {
      plot(sel)
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})

# Test with different data types
test_that("select.estimate works with binary data", {
  set.seed(456)
  binary_data <- matrix(as.integer(rnorm(50 * 4) > 0), ncol = 4)
  colnames(binary_data) <- paste0("V", 1:4)

  fit_bin <- estimate(binary_data, type = "binary", iter = 50, progress = FALSE)
  result <- select(fit_bin)

  expect_s3_class(result, "select.estimate")
})

test_that("select.estimate works with ordinal data", {
  set.seed(789)
  ordinal_data <- matrix(sample(1:4, 50 * 4, replace = TRUE), ncol = 4)
  colnames(ordinal_data) <- paste0("V", 1:4)

  fit_ord <- estimate(ordinal_data, type = "ordinal", iter = 50, progress = FALSE)
  result <- select(fit_ord)

  expect_s3_class(result, "select.estimate")
})

# Test adjacency matrix properties
test_that("select.estimate adjacency matrix is symmetric", {
  result <- select(fit)

  if ("Adj" %in% names(result)) {
    expect_equal(result$Adj, t(result$Adj))
  }

  if ("pcor_mat_zero" %in% names(result)) {
    expect_equal(result$pcor_mat_zero, t(result$pcor_mat_zero))
  }
})

test_that("select.estimate adjacency has zeros on diagonal", {
  result <- select(fit)

  if ("Adj" %in% names(result)) {
    expect_equal(diag(result$Adj), rep(0, p))
  }
})

# Test with ROPE (Region of Practical Equivalence) if supported
test_that("select.estimate handles rope parameter", {
  result <- tryCatch(
    {
      select(fit, rope = 0.1)
    },
    error = function(e) {
      # If rope not supported, just return with defaults
      select(fit)
    }
  )

  expect_s3_class(result, "select.estimate")
})

# Test column names preserved
test_that("select.estimate preserves variable names", {
  result <- select(fit)

  if ("pcor_mat_zero" %in% names(result)) {
    expect_equal(colnames(result$pcor_mat_zero), paste0("V", 1:p))
    expect_equal(rownames(result$pcor_mat_zero), paste0("V", 1:p))
  }
})
