library(testthat)
library(BGGM)

# ============================================
# Tests for predicted_probability()
# ============================================

test_that("predicted_probability returns correct structure", {
  skip_on_cran()

  set.seed(123)
  # Create ordinal-like data
  n <- 100
  p <- 5
  Y <- matrix(sample(0:3, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)
  Y <- as.data.frame(Y)

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)
  pred <- posterior_predict(fit, iter = 50, progress = FALSE)

  prob <- predicted_probability(pred, Y = Y, outcome = "V1")

  expect_true("collect" %in% names(prob))
  expect_true(is.matrix(prob$collect))
})

test_that("predicted_probability probabilities sum to 1", {
  skip_on_cran()

  set.seed(123)
  n <- 100
  p <- 5
  Y <- matrix(sample(0:3, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)
  Y <- as.data.frame(Y)

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)
  pred <- posterior_predict(fit, iter = 50, progress = FALSE)

  prob <- predicted_probability(pred, Y = Y, outcome = "V1")

  # Each row should sum to 1
  row_sums <- rowSums(prob$collect)
  expect_true(all(abs(row_sums - 1) < 1e-10))
})

test_that("predicted_probability returns probabilities in [0,1]", {
  skip_on_cran()

  set.seed(123)
  n <- 100
  p <- 5
  Y <- matrix(sample(0:3, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)
  Y <- as.data.frame(Y)

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)
  pred <- posterior_predict(fit, iter = 50, progress = FALSE)

  prob <- predicted_probability(pred, Y = Y, outcome = "V1")

  expect_true(all(prob$collect >= 0))
  expect_true(all(prob$collect <= 1))
})

test_that("predicted_probability has correct number of columns for categories", {
  skip_on_cran()

  set.seed(123)
  n <- 100
  p <- 5
  # Create data with known number of categories (0, 1, 2)
  Y <- matrix(sample(0:2, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)
  Y <- as.data.frame(Y)

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)
  pred <- posterior_predict(fit, iter = 50, progress = FALSE)

  prob <- predicted_probability(pred, Y = Y, outcome = "V1")

  # Should have 3 columns (for categories 0, 1, 2)
  expect_equal(ncol(prob$collect), 3)
})

test_that("predicted_probability has correct number of rows for iterations", {
  skip_on_cran()

  set.seed(123)
  n <- 100
  p <- 5
  iter <- 30

  Y <- matrix(sample(0:2, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)
  Y <- as.data.frame(Y)

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)
  pred <- posterior_predict(fit, iter = iter, progress = FALSE)

  prob <- predicted_probability(pred, Y = Y, outcome = "V1")

  expect_equal(nrow(prob$collect), iter)
})

test_that("predicted_probability errors for wrong class", {
  set.seed(123)
  Y <- matrix(rnorm(100), 20, 5)
  colnames(Y) <- paste0("V", 1:5)

  # Should error when passed non-posterior_predict object
  expect_error(
    predicted_probability(Y, Y = Y, outcome = "V1")
  )
})

test_that("predicted_probability conditional on other nodes", {
  skip_on_cran()

  set.seed(123)
  n <- 200
  p <- 5

  Y <- matrix(sample(0:2, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)
  Y <- as.data.frame(Y)

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)
  pred <- posterior_predict(fit, iter = 50, progress = FALSE)

  # Conditional probability
  prob <- predicted_probability(pred, Y = Y, outcome = "V1", V2 = 0)

  expect_true("collect" %in% names(prob))
  # Probabilities should still sum to 1
  row_sums <- rowSums(prob$collect)
  expect_true(all(abs(row_sums - 1) < 1e-10))
})

test_that("predicted_probability conditional on multiple nodes", {
  skip_on_cran()

  set.seed(123)
  n <- 500  # Need more data for multiple conditions
  p <- 5

  Y <- matrix(sample(0:2, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)
  Y <- as.data.frame(Y)

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)
  pred <- posterior_predict(fit, iter = 50, progress = FALSE)

  # Conditional on two nodes
  prob <- predicted_probability(pred, Y = Y, outcome = "V1", V2 = 0, V3 = 1)

  expect_true("collect" %in% names(prob))
  expect_true("sub_sets" %in% names(prob))
})

test_that("predicted_probability column names match unique values", {
  skip_on_cran()

  set.seed(123)
  n <- 100
  p <- 5

  # Create data with specific values (1, 2, 3, 4)
  Y <- matrix(sample(1:4, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)
  Y <- as.data.frame(Y)

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)
  pred <- posterior_predict(fit, iter = 50, progress = FALSE)

  prob <- predicted_probability(pred, Y = Y, outcome = "V1")

  expected_colnames <- as.character(sort(unique(Y$V1)))
  expect_equal(colnames(prob$collect), expected_colnames)
})
