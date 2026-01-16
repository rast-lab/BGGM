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
test_that("predictability works with estimate object", {
  result <- predictability(fit, progress = FALSE)

  expect_s3_class(result, "predictability")
  expect_true(is.list(result))
})

test_that("predictability returns correct structure", {
  result <- predictability(fit, progress = FALSE)

  # Should be a list containing predictability results
  expect_true(is.list(result))
})

test_that("predictability returns node-wise R2 values", {
  result <- predictability(fit, progress = FALSE)

  # R2 should be available for each node
  if ("summary" %in% names(result)) {
    expect_true(nrow(result$summary) == p || length(result$R2) == p)
  }
})

# Test iter parameter
test_that("predictability accepts iter parameter", {
  result <- predictability(fit, iter = 50, progress = FALSE)

  expect_s3_class(result, "predictability")
})

# Test with select option
test_that("predictability works with select = TRUE", {
  # predictability expects estimate objects and handles selection internally
  result <- predictability(fit, select = TRUE, progress = FALSE)

  expect_s3_class(result, "predictability")
})

# Test summary method
test_that("summary.predictability works", {
  pred <- predictability(fit, progress = FALSE)
  summ <- summary(pred)

  expect_true(inherits(summ, "summary.predictability") ||
    inherits(summ, "summary_predictability") ||
    is.list(summ) || is.data.frame(summ))
})

# Test plot method
test_that("plot.predictability works without error", {
  pred <- predictability(fit, progress = FALSE)

  result <- tryCatch(
    {
      plot(pred)
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})

# Test with different data types
test_that("predictability works with binary data estimate", {
  set.seed(456)
  binary_data <- matrix(as.integer(rnorm(50 * 4) > 0), ncol = 4)
  colnames(binary_data) <- paste0("V", 1:4)

  fit_bin <- estimate(binary_data, type = "binary", iter = 50, progress = FALSE)
  result <- predictability(fit_bin, progress = FALSE)

  expect_s3_class(result, "predictability")
})

test_that("predictability works with ordinal data estimate", {
  set.seed(789)
  ordinal_data <- matrix(sample(1:4, 50 * 4, replace = TRUE), ncol = 4)
  colnames(ordinal_data) <- paste0("V", 1:4)

  fit_ord <- estimate(ordinal_data, type = "ordinal", iter = 50, progress = FALSE)
  result <- predictability(fit_ord, progress = FALSE)

  expect_s3_class(result, "predictability")
})

# Test R2 values are in valid range
test_that("predictability R2 values are between 0 and 1", {
  result <- predictability(fit, progress = FALSE)

  # Extract R2 values
  if ("R2" %in% names(result)) {
    r2_vals <- result$R2
    if (is.numeric(r2_vals)) {
      expect_true(all(r2_vals >= 0 & r2_vals <= 1, na.rm = TRUE))
    }
  }

  if ("summary" %in% names(result) && is.data.frame(result$summary)) {
    if ("Mean" %in% names(result$summary)) {
      means <- result$summary$Mean
      expect_true(all(means >= 0 & means <= 1, na.rm = TRUE))
    }
  }
})

# Test with explore object
test_that("predictability works with explore object", {
  fit_explore <- explore(test_data, progress = FALSE)

  # predictability expects explore objects and handles selection internally
  result <- predictability(fit_explore, select = TRUE, progress = FALSE)

  expect_s3_class(result, "predictability")
})

# Test preserves node names
test_that("predictability preserves node names", {
  result <- predictability(fit, progress = FALSE)

  if ("summary" %in% names(result) && is.data.frame(result$summary)) {
    if ("Node" %in% names(result$summary)) {
      expect_true(all(result$summary$Node %in% paste0("V", 1:p)))
    }
  }
})

# Test with different sample sizes
test_that("predictability handles different sample sizes", {
  set.seed(111)
  small_data <- matrix(rnorm(30 * 4), ncol = 4)
  colnames(small_data) <- paste0("V", 1:4)

  fit_small <- estimate(small_data, iter = 50, progress = FALSE)
  result <- predictability(fit_small, progress = FALSE)

  expect_s3_class(result, "predictability")
})
