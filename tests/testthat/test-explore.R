library(testthat)
library(BGGM)

# Test data setup
set.seed(123)
test_data_cont <- matrix(rnorm(100), ncol = 5)
colnames(test_data_cont) <- paste0("V", 1:5)

test_data_binary <- matrix(as.integer(rnorm(100) > 0), ncol = 5)
colnames(test_data_binary) <- paste0("V", 1:5)

test_data_ordinal <- matrix(sample(1:4, 100, replace = TRUE), ncol = 5)
colnames(test_data_ordinal) <- paste0("V", 1:5)

# Basic functionality tests
test_that("explore with default parameters works for continuous data", {
  result <- explore(test_data_cont, progress = FALSE)

  expect_s3_class(result, c("BGGM", "explore", "default"))
  expect_true(is.list(result))
  expect_true("pcor_mat" %in% names(result))
})

test_that("explore returns correct class for continuous type", {
  result <- explore(test_data_cont, type = "continuous", progress = FALSE)

  expect_s3_class(result, "explore")
  expect_identical(result$type, "continuous")
})

test_that("explore returns correct structure", {
  result <- explore(test_data_cont, progress = FALSE)

  expect_true(is.list(result))
  expect_true("pcor_mat" %in% names(result))
  expect_true("post_samp" %in% names(result))
  expect_true(is.matrix(result$pcor_mat))
  expect_equal(dim(result$pcor_mat), c(5, 5))
})

# Test different data types
test_that("explore handles binary data", {
  result <- explore(test_data_binary, type = "binary", iter = 10, progress = FALSE)

  expect_s3_class(result, "explore")
  expect_identical(result$type, "binary")
  expect_equal(dim(result$pcor_mat), c(5, 5))
})

test_that("explore handles ordinal data", {
  result <- explore(test_data_ordinal, type = "ordinal", iter = 10, progress = FALSE)

  expect_s3_class(result, "explore")
  expect_identical(result$type, "ordinal")
  expect_equal(dim(result$pcor_mat), c(5, 5))
})

test_that("explore handles mixed data type", {
  set.seed(456)
  mixed_data <- data.frame(
    cont = rnorm(30),
    bin = as.integer(rnorm(30) > 0),
    ord = sample(1:3, 30, replace = TRUE)
  )

  result <- explore(mixed_data, type = "mixed", iter = 10, progress = FALSE)

  expect_s3_class(result, "explore")
  expect_identical(result$type, "mixed")
})

# Test with formula (control variables)
test_that("explore handles formula for control variables", {
  set.seed(789)
  Y <- matrix(rnorm(100), ncol = 4)
  colnames(Y) <- paste0("node", 1:4)
  control_var <- rnorm(25)
  dat <- data.frame(Y, control = control_var)

  result <- explore(dat, formula = ~control, progress = FALSE)

  expect_s3_class(result, "explore")
  expect_equal(result$p, 4)
  expect_true("X" %in% names(result))
})

# Test error handling
test_that("explore rejects invalid type", {
  expect_error(explore(test_data_cont, type = "invalid", progress = FALSE))
})

# Test prior specification
test_that("explore accepts prior_sd parameter", {
  result <- explore(test_data_cont, prior_sd = 0.5, progress = FALSE)

  expect_s3_class(result, "explore")
})

# Note: analytic = TRUE is not available for explore (only for estimate)

# Test summary method
test_that("summary.explore returns expected output", {
  fit <- explore(test_data_cont, progress = FALSE)
  summ <- summary(fit)

  expect_s3_class(summ, "summary_explore")
  expect_true(is.data.frame(summ$dat_results))
})

# Test plot method
test_that("plot.summary_explore works without error", {
  fit <- explore(test_data_cont, progress = FALSE)
  summ <- summary(fit)

  expect_silent(plot(summ))
})

# Test posterior samples structure
test_that("explore posterior samples have correct structure", {
  result <- explore(test_data_cont, iter = 100, progress = FALSE)

  expect_true(is.array(result$post_samp$pcors))
  # Posterior samples should have positive number of samples
  expect_true(dim(result$post_samp$pcors)[3] > 0)
})

# Test with subset of data
test_that("explore handles small datasets", {
  small_data <- matrix(rnorm(40), ncol = 4)
  colnames(small_data) <- paste0("V", 1:4)

  result <- explore(small_data, progress = FALSE)

  expect_s3_class(result, "explore")
  expect_equal(dim(result$pcor_mat), c(4, 4))
})
