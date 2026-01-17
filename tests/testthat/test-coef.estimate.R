library(testthat)
library(BGGM)

# ============================================
# Tests for coef.estimate() and coef.explore()
# ============================================

test_that("coef.estimate returns correct class", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  coefs <- coef(fit, progress = FALSE)

  expect_s3_class(coefs, "BGGM")
  expect_s3_class(coefs, "coef")
})

test_that("coef.estimate returns expected components", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  coefs <- coef(fit, progress = FALSE)

  expect_true("betas" %in% names(coefs))
  expect_true("object" %in% names(coefs))
})

test_that("coef.estimate betas has correct length", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  coefs <- coef(fit, progress = FALSE)

  # Should have p betas (one per node)
  expect_equal(length(coefs$betas), 5)
})

test_that("coef.estimate betas have correct dimensions", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]
  iter <- 50

  fit <- estimate(Y, iter = 100, progress = FALSE)
  coefs <- coef(fit, iter = iter, progress = FALSE)

  # Each beta matrix should be (p-1) x iter
  expect_equal(nrow(coefs$betas[[1]]), iter)
  expect_equal(ncol(coefs$betas[[1]]), 4)  # p - 1
})

test_that("coef.explore returns correct class", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  coefs <- coef(fit, progress = FALSE)

  expect_s3_class(coefs, "BGGM")
  expect_s3_class(coefs, "coef")
})

test_that("coef.explore betas has correct length", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  coefs <- coef(fit, progress = FALSE)

  expect_equal(length(coefs$betas), 5)
})

test_that("coef respects iter parameter", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)

  coefs_50 <- coef(fit, iter = 50, progress = FALSE)
  coefs_30 <- coef(fit, iter = 30, progress = FALSE)

  expect_equal(nrow(coefs_50$betas[[1]]), 50)
  expect_equal(nrow(coefs_30$betas[[1]]), 30)
})

test_that("coef errors for non-default object", {
  set.seed(123)
  Y <- matrix(rnorm(100), 20, 5)

  expect_error(
    coef(Y, progress = FALSE)
  )
})

test_that("coef print method works", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  coefs <- coef(fit, progress = FALSE)

  output <- capture.output(print(coefs))

  expect_true(length(output) > 0)
  expect_true(any(grepl("BGGM", output)))
  expect_true(any(grepl("Coefficients", output)))
})

test_that("summary.coef returns correct class", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  coefs <- coef(fit, progress = FALSE)
  summ <- summary(coefs)

  expect_s3_class(summ, "BGGM")
  expect_s3_class(summ, "summary.coef")
})

test_that("summary.coef returns expected components", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  coefs <- coef(fit, progress = FALSE)
  summ <- summary(coefs)

  expect_true("summaries" %in% names(summ))
  expect_true("object" %in% names(summ))
})

test_that("summary.coef summaries has correct length", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  coefs <- coef(fit, progress = FALSE)
  summ <- summary(coefs)

  expect_equal(length(summ$summaries), 5)
})

test_that("summary.coef summaries has correct columns", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  coefs <- coef(fit, progress = FALSE)
  summ <- summary(coefs)

  expected_cols <- c("post_mean", "post_sd", "post_lb", "post_ub")
  expect_equal(colnames(summ$summaries[[1]]), expected_cols)
})

test_that("summary.coef respects cred parameter", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  coefs <- coef(fit, progress = FALSE)

  summ_95 <- summary(coefs, cred = 0.95)
  summ_90 <- summary(coefs, cred = 0.90)

  # Different cred should give different intervals
  expect_s3_class(summ_95, "summary.coef")
  expect_s3_class(summ_90, "summary.coef")
})

test_that("summary.coef print method works", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  coefs <- coef(fit, progress = FALSE)
  summ <- summary(coefs)

  output <- capture.output(print(summ))

  expect_true(length(output) > 0)
  expect_true(any(grepl("BGGM", output)))
})

test_that("coef with progress = TRUE works", {
  set.seed(123)
  Y <- BGGM::bfi[1:30, 1:3]

  fit <- estimate(Y, iter = 100, progress = FALSE)

  expect_no_error({
    capture.output({
      coefs <- coef(fit, progress = TRUE)
    })
  })
})

test_that("coef works with binary data", {
  set.seed(123)
  Y <- matrix(rbinom(150, 1, 0.5), 50, 3)
  colnames(Y) <- paste0("V", 1:3)

  fit <- estimate(Y, type = "binary", iter = 100, progress = FALSE)
  coefs <- coef(fit, progress = FALSE)

  expect_s3_class(coefs, "coef")
  expect_equal(length(coefs$betas), 3)
})

test_that("coef works with ordinal data", {
  set.seed(123)
  Y <- matrix(sample(1:5, 150, replace = TRUE), 50, 3)
  colnames(Y) <- paste0("V", 1:3)

  fit <- estimate(Y, type = "ordinal", iter = 100, progress = FALSE)
  coefs <- coef(fit, progress = FALSE)

  expect_s3_class(coefs, "coef")
})

test_that("coef betas contain finite values", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  coefs <- coef(fit, progress = FALSE)

  for (i in 1:5) {
    expect_true(all(is.finite(coefs$betas[[i]])))
  }
})

