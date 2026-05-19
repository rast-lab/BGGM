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

  expect_true(all(c("scores", "type", "metric") %in% names(result)))
})

test_that("predictability returns node-wise R2 values", {
  result <- predictability(fit, progress = FALSE)

  expect_true("scores" %in% names(result))
  expect_equal(length(result$scores), p)
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
test_that("plot.predictability returns a ggplot", {
  pred <- predictability(fit, progress = FALSE)
  plt  <- plot(pred)
  expect_true(inherits(plt, "ggplot"))
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
  summ   <- summary(result)$summary

  expect_true(all(summ$Post.mean >= 0 & summ$Post.mean <= 1))
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
  summ   <- summary(result)$summary

  expect_equal(summ$Node, paste0("V", 1:p))
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

# ============================================
# Additional tests for coverage
# ============================================

test_that("predictability works with var_estimate object", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  result <- predictability(fit, progress = FALSE)

  expect_s3_class(result, "predictability")
})

test_that("predictability var_estimate warns for select = TRUE", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)

  expect_warning(
    result <- predictability(fit, select = TRUE, progress = FALSE),
    "'select' not implemented"
  )
  expect_s3_class(result, "predictability")
})

test_that("summary.predictability returns correct structure", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  pred <- predictability(fit, progress = FALSE)
  summ <- summary(pred)

  expect_true("summary" %in% names(summ))
  expect_true(is.data.frame(summ$summary))
})

test_that("summary.predictability has correct columns", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  pred <- predictability(fit, progress = FALSE)
  summ <- summary(pred)

  expected_cols <- c("Node", "Post.mean", "Post.sd", "Cred")
  expect_true(all(expected_cols[1:3] %in% colnames(summ$summary)))
})

test_that("summary.predictability respects cred parameter", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  pred <- predictability(fit, progress = FALSE)

  summ_95 <- summary(pred, cred = 0.95)
  summ_90 <- summary(pred, cred = 0.90)

  expect_true(is.data.frame(summ_95$summary))
  expect_true(is.data.frame(summ_90$summary))
})

test_that("summary.predictability without column names", {
  set.seed(123)
  Y <- as.matrix(BGGM::bfi[1:50, 1:5])
  colnames(Y) <- NULL

  fit <- estimate(Y, iter = 100, progress = FALSE)
  pred <- predictability(fit, progress = FALSE)
  summ <- summary(pred)

  expect_true(is.data.frame(summ$summary))
})

test_that("plot.predictability error_bar type works", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  pred <- predictability(fit, progress = FALSE)

  plt <- plot(pred, type = "error_bar")

  expect_true(inherits(plt, "ggplot"))
})

test_that("plot.predictability ridgeline type works", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  pred <- predictability(fit, progress = FALSE)

  plt <- plot(pred, type = "ridgeline")

  expect_true(inherits(plt, "ggplot"))
})

test_that("plot.predictability respects cred parameter", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  pred <- predictability(fit, progress = FALSE)

  plt <- plot(pred, cred = 0.90)

  expect_true(inherits(plt, "ggplot"))
})

test_that("plot.predictability respects color parameter", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  pred <- predictability(fit, progress = FALSE)

  plt <- plot(pred, color = "red")

  expect_true(inherits(plt, "ggplot"))
})

test_that("plot.predictability respects size parameter", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  pred <- predictability(fit, progress = FALSE)

  plt <- plot(pred, size = 3)

  expect_true(inherits(plt, "ggplot"))
})

test_that("plot.predictability errors for invalid type", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  pred <- predictability(fit, progress = FALSE)

  expect_error(
    plot(pred, type = "invalid"),
    "type not supported"
  )
})

test_that("predictability with mixed data type", {
  set.seed(123)
  Y <- BGGM::ptsd[1:50, 1:5]

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)
  pred <- predictability(fit, progress = FALSE)

  expect_s3_class(pred, "predictability")
})

test_that("predictability select = TRUE with estimate", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  pred <- predictability(fit, select = TRUE, cred = 0.50, progress = FALSE)

  expect_s3_class(pred, "predictability")
})

test_that("predictability select = TRUE with explore", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  pred <- predictability(fit, select = TRUE, BF_cut = 1, progress = FALSE)

  expect_s3_class(pred, "predictability")
})
