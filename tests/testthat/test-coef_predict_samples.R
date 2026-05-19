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

  expect_true(inherits(result, "BGGM") && inherits(result, "coef"))
  expect_true("betas" %in% names(result))
})

test_that("coef.estimate betas is a list with one entry per node", {
  result <- coef(fit)

  expect_true(is.list(result$betas))
  expect_equal(length(result$betas), p)
})

test_that("coef.estimate beta matrices have correct dimensions", {
  result <- coef(fit)

  # Each beta matrix: iter rows, (p-1) columns (one predictor per other node)
  expect_true(is.matrix(result$betas[[1]]))
  expect_equal(ncol(result$betas[[1]]), p - 1)
  expect_equal(nrow(result$betas[[1]]), fit$iter)
})

test_that("coef.estimate betas differ across nodes", {
  result <- coef(fit)

  # Different nodes should have different posterior samples
  expect_false(isTRUE(all.equal(result$betas[[1]], result$betas[[2]])))
})

# Test that print produces non-empty output with expected headers
test_that("coef.estimate print produces coefficient output", {
  result <- coef(fit)

  output <- capture.output(print(result))
  expect_true(any(grepl("Coefficients", output)))
  expect_true(any(grepl("BGGM", output)))
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

  expect_true(is.matrix(result))
  expect_equal(nrow(result), fit$iter)
})

test_that("posterior_samples returns one column per unique edge", {
  result <- posterior_samples(fit)

  n_edges <- p * (p - 1) / 2
  expect_equal(ncol(result), n_edges)
})

test_that("posterior_samples works with explore object", {
  fit_explore <- explore(test_data, progress = FALSE)
  result <- posterior_samples(fit_explore)

  expect_true(is.matrix(result))
  expect_equal(ncol(result), p * (p - 1) / 2)
})

test_that("posterior_samples values are in [-1, 1]", {
  result <- posterior_samples(fit)

  expect_true(all(result >= -1 & result <= 1))
})

# ============================================
# Tests for regression_summary
# ============================================

test_that("regression_summary errors without a formula-based estimate", {
  expect_error(regression_summary(fit))
})

test_that("regression_summary works with formula-based estimate", {
  Y <- as.data.frame(BGGM::bfi[1:50, c("A1", "A2", "A3", "gender")])
  fit_f <- estimate(Y, formula = ~gender, type = "continuous",
                    iter = 50, progress = FALSE)
  rs <- regression_summary(fit_f)

  expect_s3_class(rs, "regression_summary")
  expect_true("reg_summary" %in% names(rs))
  expect_true("resid_cor"   %in% names(rs))
  expect_equal(length(rs$reg_summary), 3)  # 3 outcome variables (A1, A2, A3)
})

# ============================================
# Tests with different data types
# ============================================

test_that("coef works with binary data estimate", {
  set.seed(333)
  binary_data <- matrix(as.integer(rnorm(50 * 4) > 0), ncol = 4)
  colnames(binary_data) <- paste0("V", 1:4)

  fit_bin <- estimate(binary_data, type = "binary", iter = 50, progress = FALSE)
  result  <- coef(fit_bin)

  expect_true(inherits(result, "BGGM") && inherits(result, "coef"))
  expect_equal(length(result$betas), 4)
})

test_that("coef works with ordinal data estimate", {
  set.seed(444)
  ordinal_data <- matrix(sample(1:4, 50 * 4, replace = TRUE), ncol = 4)
  colnames(ordinal_data) <- paste0("V", 1:4)

  fit_ord <- estimate(ordinal_data, type = "ordinal", iter = 50, progress = FALSE)
  result  <- coef(fit_ord)

  expect_true(inherits(result, "BGGM") && inherits(result, "coef"))
  expect_equal(length(result$betas), 4)
})

test_that("posterior_samples row count matches iterations", {
  fit_50  <- estimate(test_data, iter = 50,  progress = FALSE)
  fit_100 <- estimate(test_data, iter = 100, progress = FALSE)

  expect_equal(nrow(posterior_samples(fit_50)),  50)
  expect_equal(nrow(posterior_samples(fit_100)), 100)
})

# ============================================
# Tests for posterior_predict
# ============================================

test_that("posterior_predict errors on continuous estimate", {
  # posterior_predict only supports binary/ordinal/mixed types
  expect_error(posterior_predict(fit), "type must be")
})

test_that("posterior_predict works with binary estimate", {
  set.seed(777)
  binary_data <- matrix(as.integer(rnorm(60 * 4) > 0), ncol = 4)
  colnames(binary_data) <- paste0("V", 1:4)

  fit_bin <- estimate(binary_data, type = "binary", iter = 100, progress = FALSE)
  result  <- posterior_predict(fit_bin, iter = 20, progress = FALSE)

  expect_true(is.array(result))
  expect_equal(dim(result)[1], 60)  # observations
  expect_equal(dim(result)[3], 20)  # predictive draws
})

test_that("posterior_predict returns binary values for binary model", {
  set.seed(888)
  binary_data <- matrix(as.integer(rnorm(60 * 4) > 0), ncol = 4)
  colnames(binary_data) <- paste0("V", 1:4)

  fit_bin <- estimate(binary_data, type = "binary", iter = 100, progress = FALSE)
  result  <- posterior_predict(fit_bin, iter = 10, progress = FALSE)

  expect_true(all(result %in% c(0, 1)))
})
