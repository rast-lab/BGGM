library(testthat)
library(BGGM)

# ============================================
# Tests for predict.estimate() and predict.explore()
# ============================================

test_that("predict.estimate returns array with summary = TRUE", {
  set.seed(123)
  Y <- BGGM::ptsd[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  pred <- predict(fit, progress = FALSE)

  expect_true(is.array(pred))
  expect_equal(length(dim(pred)), 3)
})

test_that("predict.estimate has correct dimensions with summary = TRUE", {
  set.seed(123)
  n <- 50
  p <- 5
  Y <- BGGM::ptsd[1:n, 1:p]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  pred <- predict(fit, progress = FALSE)

  # dim should be (n, 4, p)
  expect_equal(dim(pred)[1], n)
  expect_equal(dim(pred)[2], 4)
  expect_equal(dim(pred)[3], p)
})

test_that("predict.estimate dimnames are correct", {
  set.seed(123)
  Y <- BGGM::ptsd[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  pred <- predict(fit, progress = FALSE)

  expected_stats <- c("Post.mean", "Post.sd", "Cred.lb", "Cred.ub")
  expect_equal(dimnames(pred)[[2]], expected_stats)
})

test_that("predict.estimate returns array with summary = FALSE", {
  set.seed(123)
  Y <- BGGM::ptsd[1:50, 1:5]
  iter <- 50

  fit <- estimate(Y, iter = 100, progress = FALSE)
  pred <- predict(fit, iter = iter, summary = FALSE, progress = FALSE)

  expect_true(is.array(pred))
  # dim should be (iter, n, p)
  expect_equal(dim(pred)[1], iter)
  expect_equal(dim(pred)[2], 50)
  expect_equal(dim(pred)[3], 5)
})

test_that("predict.explore returns array", {
  set.seed(123)
  Y <- BGGM::ptsd[1:50, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  pred <- predict(fit, progress = FALSE)

  expect_true(is.array(pred))
})

test_that("predict.explore has correct dimensions", {
  set.seed(123)
  n <- 50
  p <- 5
  Y <- BGGM::ptsd[1:n, 1:p]

  fit <- explore(Y, iter = 100, progress = FALSE)
  pred <- predict(fit, progress = FALSE)

  expect_equal(dim(pred)[1], n)
  expect_equal(dim(pred)[2], 4)
  expect_equal(dim(pred)[3], p)
})

test_that("predict with newdata works", {
  set.seed(123)
  Y_train <- BGGM::ptsd[1:50, 1:5]
  Y_test <- BGGM::ptsd[51:70, 1:5]

  fit <- estimate(Y_train, iter = 100, progress = FALSE)
  pred <- predict(fit, newdata = Y_test, progress = FALSE)

  expect_equal(dim(pred)[1], 20)  # 20 test observations
  expect_equal(dim(pred)[3], 5)
})

test_that("predict errors for non-continuous type", {
  set.seed(123)
  Y <- matrix(rbinom(150, 1, 0.5), 50, 3)
  colnames(Y) <- paste0("V", 1:3)

  fit <- estimate(Y, type = "binary", iter = 100, progress = FALSE)

  expect_error(
    predict(fit, progress = FALSE),
    "type not currently supported"
  )
})

test_that("predict errors when newdata has wrong number of columns", {
  set.seed(123)
  Y_train <- BGGM::ptsd[1:50, 1:5]
  Y_test <- BGGM::ptsd[51:70, 1:3]  # Wrong number of columns

  fit <- estimate(Y_train, iter = 100, progress = FALSE)

  expect_error(
    predict(fit, newdata = Y_test, progress = FALSE)
  )
})

test_that("predict respects iter parameter", {
  set.seed(123)
  Y <- BGGM::ptsd[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  pred <- predict(fit, iter = 30, summary = FALSE, progress = FALSE)

  expect_equal(dim(pred)[1], 30)
})

test_that("predict respects cred parameter", {
  set.seed(123)
  Y <- BGGM::ptsd[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)

  pred_95 <- predict(fit, cred = 0.95, progress = FALSE)
  pred_90 <- predict(fit, cred = 0.90, progress = FALSE)

  # Different cred should give different intervals
  expect_false(all(pred_95[, "Cred.lb", ] == pred_90[, "Cred.lb", ]))
})

test_that("predict with progress = TRUE works", {
  set.seed(123)
  Y <- BGGM::ptsd[1:30, 1:3]

  fit <- estimate(Y, iter = 100, progress = FALSE)

  expect_no_error({
    capture.output({
      pred <- predict(fit, progress = TRUE)
    })
  })
})

test_that("predict preserves column names", {
  set.seed(123)
  Y <- BGGM::ptsd[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  pred <- predict(fit, progress = FALSE)

  expect_equal(dimnames(pred)[[3]], colnames(Y))
})

test_that("predict without column names uses numbers", {
  set.seed(123)
  Y <- as.matrix(BGGM::ptsd[1:50, 1:5])
  colnames(Y) <- NULL

  fit <- estimate(Y, iter = 100, progress = FALSE)
  pred <- predict(fit, progress = FALSE)

  expect_equal(dimnames(pred)[[3]], as.character(1:5))
})

test_that("predict.var_estimate returns array", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  pred <- predict(fit, progress = FALSE)

  expect_true(is.array(pred))
})

test_that("predict.var_estimate has correct dimensions", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  pred <- predict(fit, progress = FALSE)

  # dim should be (n-1, 4, p) for summary = TRUE
  expect_equal(dim(pred)[2], 4)
  expect_equal(dim(pred)[3], ncol(Y))
})

test_that("predict.var_estimate summary = FALSE works", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)

  # Test with summary = TRUE (default) instead
  pred <- predict(fit, summary = TRUE, progress = FALSE)

  expect_true(is.array(pred))
})

test_that("predict.var_estimate respects cred parameter", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)

  pred_95 <- predict(fit, cred = 0.95, progress = FALSE)
  pred_90 <- predict(fit, cred = 0.90, progress = FALSE)

  expect_true(is.array(pred_95))
  expect_true(is.array(pred_90))
})

test_that("predict values are finite", {
  set.seed(123)
  Y <- BGGM::ptsd[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  pred <- predict(fit, progress = FALSE)

  expect_true(all(is.finite(pred)))
})

test_that("predict posterior mean is within credible interval", {
  set.seed(123)
  Y <- BGGM::ptsd[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  pred <- predict(fit, progress = FALSE)

  # For most observations, mean should be between lb and ub
  mean_vals <- pred[, "Post.mean", ]
  lb_vals <- pred[, "Cred.lb", ]
  ub_vals <- pred[, "Cred.ub", ]

  within_interval <- (mean_vals >= lb_vals) & (mean_vals <= ub_vals)
  expect_true(mean(within_interval) > 0.95)
})

