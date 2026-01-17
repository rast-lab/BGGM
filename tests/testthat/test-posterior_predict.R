library(testthat)
library(BGGM)

# ============================================
# Tests for posterior_predict()
# ============================================

test_that("posterior_predict returns correct class", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(sample(0:3, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)
  pred <- posterior_predict(fit, iter = 20, progress = FALSE)

  expect_s3_class(pred, "posterior_predict")
  expect_true(inherits(pred, "array"))
})

test_that("posterior_predict returns correct dimensions", {
  set.seed(123)
  n <- 50
  p <- 3
  iter <- 20
  Y <- matrix(sample(0:3, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)
  pred <- posterior_predict(fit, iter = iter, progress = FALSE)

  expect_equal(dim(pred)[1], n)
  expect_equal(dim(pred)[2], p)
  expect_equal(dim(pred)[3], iter)
})

test_that("posterior_predict preserves column names", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(sample(0:3, n * p, replace = TRUE), n, p)
  colnames(Y) <- c("A", "B", "C")

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)
  pred <- posterior_predict(fit, iter = 20, progress = FALSE)

  expect_equal(dimnames(pred)[[2]], c("A", "B", "C"))
})

test_that("posterior_predict errors for non-estimate objects", {
  set.seed(123)
  Y <- matrix(rnorm(100), 20, 5)

  expect_error(
    posterior_predict(Y, iter = 20),
    "must be of class"
  )
})

test_that("posterior_predict errors for continuous type", {
  set.seed(123)
  Y <- matrix(rnorm(150), 30, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- estimate(Y, iter = 100, type = "continuous", progress = FALSE)

  expect_error(
    posterior_predict(fit, iter = 20, progress = FALSE),
    "type must be"
  )
})

test_that("posterior_predict works with mixed type data", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(sample(1:5, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)
  pred <- posterior_predict(fit, iter = 20, progress = FALSE)

  expect_s3_class(pred, "posterior_predict")
  expect_true(all(is.finite(pred)))
})

test_that("posterior_predict returns values in original data range for mixed", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(sample(1:5, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)
  pred <- posterior_predict(fit, iter = 20, progress = FALSE)

  # Predicted values should be within the range of original data
  for (j in 1:p) {
    original_values <- unique(Y[, j])
    for (s in 1:dim(pred)[3]) {
      expect_true(all(pred[, j, s] %in% original_values))
    }
  }
})

test_that("posterior_predict works with binary type data", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(sample(0:1, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)

  fit <- estimate(as.matrix(Y), iter = 100, type = "binary", progress = FALSE)
  pred <- posterior_predict(fit, iter = 20, progress = FALSE)

  expect_s3_class(pred, "posterior_predict")
  # Binary predictions should be 0 or 1
  expect_true(all(pred %in% c(0, 1)))
})

test_that("posterior_predict binary returns only 0s and 1s", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(sample(0:1, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)

  fit <- estimate(as.matrix(Y), iter = 100, type = "binary", progress = FALSE)
  pred <- posterior_predict(fit, iter = 20, progress = FALSE)

  unique_vals <- unique(as.vector(pred))
  expect_true(all(unique_vals %in% c(0, 1)))
})

test_that("posterior_predict works with ordinal type data", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(sample(1:4, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)

  fit <- estimate(as.matrix(Y), iter = 100, type = "ordinal", progress = FALSE)
  pred <- posterior_predict(fit, iter = 20, progress = FALSE)

  expect_s3_class(pred, "posterior_predict")
})

test_that("posterior_predict ordinal returns integer categories", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(sample(1:4, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)

  fit <- estimate(as.matrix(Y), iter = 100, type = "ordinal", progress = FALSE)
  pred <- posterior_predict(fit, iter = 20, progress = FALSE)

  # Ordinal predictions should be integers
  expect_true(all(pred == floor(pred)))
})

test_that("posterior_predict works with explore objects", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(sample(0:3, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)

  fit <- explore(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)
  pred <- posterior_predict(fit, iter = 20, progress = FALSE)

  expect_s3_class(pred, "posterior_predict")
})

test_that("posterior_predict produces varied predictions across iterations", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(sample(0:3, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)
  pred <- posterior_predict(fit, iter = 20, progress = FALSE)

  # Predictions should vary across iterations (not all identical)
  iter_sums <- apply(pred, 3, sum)
  expect_true(sd(iter_sums) > 0)
})

test_that("posterior_predict with progress = TRUE works", {
  set.seed(123)
  n <- 30
  p <- 3
  Y <- matrix(sample(0:2, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)

  expect_no_error({
    capture.output({
      pred <- posterior_predict(fit, iter = 10, progress = TRUE)
    })
  })
})

test_that("posterior_predict respects iter parameter", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(sample(0:3, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)

  pred_10 <- posterior_predict(fit, iter = 10, progress = FALSE)
  pred_20 <- posterior_predict(fit, iter = 20, progress = FALSE)

  expect_equal(dim(pred_10)[3], 10)
  expect_equal(dim(pred_20)[3], 20)
})

test_that("posterior_predict maintains observation count", {
  set.seed(123)
  n <- 40
  p <- 3
  Y <- matrix(sample(0:3, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)
  pred <- posterior_predict(fit, iter = 20, progress = FALSE)

  expect_equal(dim(pred)[1], n)
})

test_that("posterior_predict maintains variable count", {
  set.seed(123)
  n <- 50
  p <- 4
  Y <- matrix(sample(0:3, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)
  pred <- posterior_predict(fit, iter = 20, progress = FALSE)

  expect_equal(dim(pred)[2], p)
})

test_that("posterior_predict can be used with predicted_probability", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(sample(0:3, n * p, replace = TRUE), n, p)
  colnames(Y) <- paste0("V", 1:p)
  Y_df <- as.data.frame(Y)

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)
  pred <- posterior_predict(fit, iter = 20, progress = FALSE)

  # Should work with predicted_probability
  prob <- predicted_probability(pred, Y = Y_df, outcome = "V1")

  expect_true("collect" %in% names(prob))
  expect_true(is.matrix(prob$collect))
})

test_that("posterior_predict with real ptsd data", {
  Y <- BGGM::ptsd[1:50, 1:3]

  fit <- estimate(as.matrix(Y), iter = 100, type = "mixed", progress = FALSE)
  pred <- posterior_predict(fit, iter = 20, progress = FALSE)

  expect_s3_class(pred, "posterior_predict")
  expect_equal(dim(pred)[1], 50)
  expect_equal(dim(pred)[2], 3)
})
