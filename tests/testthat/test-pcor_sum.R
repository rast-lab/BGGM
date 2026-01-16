library(testthat)
library(BGGM)

# ============================================
# Tests for pcor_sum() - Partial Correlation Sum
# ============================================

test_that("pcor_sum returns correct class", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- estimate(Y, iter = 100, progress = FALSE)

  sums <- pcor_sum(fit, relations = "V1--V2 + V1--V3")

  expect_s3_class(sums, "BGGM")
  expect_s3_class(sums, "pcor_sum")
})

test_that("pcor_sum returns expected components", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- estimate(Y, iter = 100, progress = FALSE)

  sums <- pcor_sum(fit, relations = "V1--V2 + V1--V3")

  expect_true("post_sums" %in% names(sums))
  expect_true("n_sums" %in% names(sums))
  expect_true("iter" %in% names(sums))
})

test_that("pcor_sum single sum has correct length", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)
  iter <- 100

  fit <- estimate(Y, iter = iter, progress = FALSE)

  sums <- pcor_sum(fit, relations = "V1--V2 + V1--V3", iter = 50)

  expect_length(sums$post_sums[[1]], 50)
})

test_that("pcor_sum comparing two sums works", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- estimate(Y, iter = 100, progress = FALSE)

  sums <- pcor_sum(fit, relations = "V1--V2 + V1--V3; V2--V3 + V2--V4")

  expect_equal(sums$n_sums, 2)
  expect_true(!is.null(sums$post_diff))
  expect_length(sums$post_sums, 2)
})

test_that("pcor_sum difference is correctly computed", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)
  iter <- 50

  fit <- estimate(Y, iter = 100, progress = FALSE)

  sums <- pcor_sum(fit, relations = "V1--V2 + V1--V3; V2--V3 + V2--V4", iter = iter)

  # Difference should be sum1 - sum2
  expected_diff <- sums$post_sums[[1]] - sums$post_sums[[2]]
  expect_equal(sums$post_diff, expected_diff)
})

test_that("pcor_sum comparing two groups works", {
  set.seed(123)
  Y1 <- matrix(rnorm(200), 40, 5)
  Y2 <- matrix(rnorm(200), 40, 5)
  colnames(Y1) <- colnames(Y2) <- paste0("V", 1:5)

  fit1 <- estimate(Y1, iter = 100, progress = FALSE)
  fit2 <- estimate(Y2, iter = 100, progress = FALSE)

  sums <- pcor_sum(fit1, fit2, relations = "V1--V2 + V1--V3")

  expect_length(sums$post_sums, 2)
  expect_true(!is.null(sums$post_diff))
})

test_that("pcor_sum errors with more than two sums", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- estimate(Y, iter = 100, progress = FALSE)

  expect_error(
    pcor_sum(fit, relations = "V1--V2; V2--V3; V3--V4"),
    "at most.*two sums"
  )
})

test_that("pcor_sum errors with wrong object class", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)

  # Should error when passed non-estimate object
  expect_error(
    pcor_sum(Y, relations = "V1--V2 + V1--V3")
  )
})

test_that("pcor_sum errors with two groups and multiple sums", {
  set.seed(123)
  Y1 <- matrix(rnorm(200), 40, 5)
  Y2 <- matrix(rnorm(200), 40, 5)
  colnames(Y1) <- colnames(Y2) <- paste0("V", 1:5)

  fit1 <- estimate(Y1, iter = 100, progress = FALSE)
  fit2 <- estimate(Y2, iter = 100, progress = FALSE)

  expect_error(
    pcor_sum(fit1, fit2, relations = "V1--V2; V2--V3"),
    "only one sum can be specified"
  )
})

test_that("pcor_sum errors with too many groups", {
  set.seed(123)
  Y1 <- matrix(rnorm(200), 40, 5)
  Y2 <- matrix(rnorm(200), 40, 5)
  Y3 <- matrix(rnorm(200), 40, 5)
  colnames(Y1) <- colnames(Y2) <- colnames(Y3) <- paste0("V", 1:5)

  fit1 <- estimate(Y1, iter = 100, progress = FALSE)
  fit2 <- estimate(Y2, iter = 100, progress = FALSE)
  fit3 <- estimate(Y3, iter = 100, progress = FALSE)

  expect_error(
    pcor_sum(fit1, fit2, fit3, relations = "V1--V2"),
    "too many groups"
  )
})

test_that("pcor_sum respects iter parameter", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- estimate(Y, iter = 100, progress = FALSE)

  sums <- pcor_sum(fit, relations = "V1--V2 + V1--V3", iter = 30)

  expect_equal(sums$iter, 30)
  expect_length(sums$post_sums[[1]], 30)
})

test_that("pcor_sum print method works", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- estimate(Y, iter = 100, progress = FALSE)

  sums <- pcor_sum(fit, relations = "V1--V2 + V1--V3")

  output <- capture.output(print(sums))

  expect_true(length(output) > 0)
  expect_true(any(grepl("BGGM", output)))
})

test_that("pcor_sum print with difference works", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- estimate(Y, iter = 100, progress = FALSE)

  sums <- pcor_sum(fit, relations = "V1--V2 + V1--V3; V2--V3 + V2--V4")

  output <- capture.output(print(sums))

  expect_true(any(grepl("Difference", output)))
})

test_that("pcor_sum plot method works for single sum", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- estimate(Y, iter = 100, progress = FALSE)

  sums <- pcor_sum(fit, relations = "V1--V2 + V1--V3")

  plots <- plot(sums)

  expect_true(inherits(plots$g1, "ggplot"))
})

test_that("pcor_sum plot method works for two sums", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- estimate(Y, iter = 100, progress = FALSE)

  sums <- pcor_sum(fit, relations = "V1--V2 + V1--V3; V2--V3 + V2--V4")

  plots <- plot(sums)

  expect_true(inherits(plots$g1, "ggplot"))
  expect_true(inherits(plots$g2, "ggplot"))
  expect_true(inherits(plots$diff, "ggplot"))
})

test_that("pcor_sum handles single edge", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- estimate(Y, iter = 100, progress = FALSE)

  # Single edge (no + sign)
  sums <- pcor_sum(fit, relations = "V1--V2")

  expect_s3_class(sums, "pcor_sum")
  expect_equal(sums$n_sums, 1)
})

test_that("pcor_sum handles spaces in relations", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- estimate(Y, iter = 100, progress = FALSE)

  # With extra spaces
  sums <- pcor_sum(fit, relations = "V1--V2  +  V1--V3")

  expect_s3_class(sums, "pcor_sum")
})
