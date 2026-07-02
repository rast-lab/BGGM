library(testthat)
library(BGGM)

# Test data setup - time series data
set.seed(123)
# Create simple time series data (50 time points, 4 variables)
n_time <- 50
p <- 4
test_ts_data <- matrix(rnorm(n_time * p), ncol = p)
colnames(test_ts_data) <- paste0("V", 1:p)

# Add some autocorrelation for realism
for (t in 2:n_time) {
  test_ts_data[t, ] <- 0.3 * test_ts_data[t - 1, ] + rnorm(p, sd = 0.9)
}

# Basic functionality tests
test_that("var_estimate works with default parameters", {
  result <- var_estimate(test_ts_data, iter = 50, progress = FALSE)

  expect_s3_class(result, c("BGGM", "var_estimate"))
  expect_true(is.list(result))
})

test_that("var_estimate returns correct structure", {
  result <- var_estimate(test_ts_data, iter = 50, progress = FALSE)

  # Should contain partial correlations (GGM part) - stored as pcor_mu
  expect_true("pcor_mu" %in% names(result))
  expect_true(is.matrix(result$pcor_mu))
  expect_equal(dim(result$pcor_mu), c(p, p))

  # Should contain beta coefficients (VAR part) - stored as beta_mu
  expect_true("beta_mu" %in% names(result))
})

test_that("var_estimate handles different sample sizes", {
  # Longer time series
  set.seed(456)
  long_ts <- matrix(rnorm(100 * 4), ncol = 4)
  colnames(long_ts) <- paste0("V", 1:4)

  result <- var_estimate(long_ts, iter = 50, progress = FALSE)

  expect_s3_class(result, "var_estimate")
})

test_that("var_estimate handles different number of variables", {
  set.seed(789)
  # 3 variables
  ts_3var <- matrix(rnorm(50 * 3), ncol = 3)
  colnames(ts_3var) <- paste0("V", 1:3)

  result <- var_estimate(ts_3var, iter = 50, progress = FALSE)

  expect_s3_class(result, "var_estimate")
  expect_equal(dim(result$pcor_mu), c(3, 3))
})

# Test prior specification
test_that("var_estimate accepts rho_sd parameter", {
  # real parameter name for the correlation prior; a nonexistent
  # `prior_sd` was silently swallowed by `...` here previously, so this
  # test always ran with default priors regardless of what it passed
  result <- var_estimate(
    test_ts_data,
    rho_sd = 0.5,
    iter = 50,
    progress = FALSE
  )

  expect_s3_class(result, "var_estimate")
})

# Test posterior mean estimates
test_that("var_estimate posterior mean estimates have correct dimensions", {
  result <- var_estimate(test_ts_data, iter = 100, progress = FALSE)

  # Check pcor_mu (posterior mean of partial correlations) exists
  expect_true("pcor_mu" %in% names(result))
  expect_true(is.matrix(result$pcor_mu))
})

# Test summary method
test_that("summary.var_estimate returns expected output", {
  fit <- var_estimate(test_ts_data, iter = 50, progress = FALSE)
  summ <- summary(fit)

  expect_true(inherits(summ, "summary_var_estimate") ||
    inherits(summ, "summary.var_estimate"))
})

# Test with named columns
test_that("var_estimate preserves column names", {
  set.seed(111)
  named_data <- matrix(rnorm(50 * 4), ncol = 4)
  colnames(named_data) <- c("anxiety", "depression", "stress", "sleep")

  result <- var_estimate(named_data, iter = 50, progress = FALSE)

  expect_s3_class(result, "var_estimate")
  expect_equal(colnames(result$pcor_mu), c("anxiety", "depression", "stress", "sleep"))
})

# Test select method for var_estimate
test_that("select works with var_estimate object", {
  fit <- var_estimate(test_ts_data, iter = 100, progress = FALSE)

  # Try selecting edges
  selected <- select(fit)

  expect_true(is.list(selected) || inherits(selected, "select.var_estimate"))
})

# Test edge cases
test_that("var_estimate handles minimum viable time series", {
  # Minimum time points needed
  set.seed(222)
  min_ts <- matrix(rnorm(20 * 4), ncol = 4)
  colnames(min_ts) <- paste0("V", 1:4)

  result <- var_estimate(min_ts, iter = 50, progress = FALSE)

  expect_s3_class(result, "var_estimate")
})

# Note: there is no plot.var_estimate method (confirmed against NAMESPACE's
# registered S3 methods), only plot.summary.var_estimate -- plot(fit)
# directly dispatches to plot.default and errors. See "plot.summary.var_estimate
# works" below for real coverage of the correct usage (plot(summary(fit))).

# ============================================
# Additional tests for coverage
# ============================================

test_that("var_estimate accepts beta_sd parameter", {
  result <- var_estimate(
    test_ts_data,
    beta_sd = 0.5,
    iter = 50,
    progress = FALSE
  )

  expect_s3_class(result, "var_estimate")
})

test_that("var_estimate accepts seed parameter", {
  result1 <- var_estimate(test_ts_data, iter = 50, seed = 123, progress = FALSE)
  result2 <- var_estimate(test_ts_data, iter = 50, seed = 123, progress = FALSE)

  expect_s3_class(result1, "var_estimate")
  expect_s3_class(result2, "var_estimate")
})

test_that("var_estimate errors with constant variable", {
  set.seed(123)
  bad_data <- test_ts_data
  bad_data[, 1] <- 1  # Constant value

  expect_error(
    var_estimate(bad_data, iter = 50, progress = FALSE)
  )
})

test_that("var_estimate handles NA values", {
  set.seed(123)
  na_data <- test_ts_data
  na_data[1, 1] <- NA

  result <- var_estimate(na_data, iter = 50, progress = FALSE)

  expect_s3_class(result, "var_estimate")
})

test_that("var_estimate print method works", {
  fit <- var_estimate(test_ts_data, iter = 50, progress = FALSE)

  output <- capture.output(print(fit))

  expect_true(length(output) > 0)
  expect_true(any(grepl("BGGM", output)))
  expect_true(any(grepl("VAR", output)))
})

test_that("summary.var_estimate returns correct structure", {
  fit <- var_estimate(test_ts_data, iter = 50, progress = FALSE)
  summ <- summary(fit)

  expect_true("pcor_results" %in% names(summ))
  expect_true("beta_results" %in% names(summ))
})

test_that("summary.var_estimate pcor_results is data.frame", {
  fit <- var_estimate(test_ts_data, iter = 50, progress = FALSE)
  summ <- summary(fit)

  expect_true(is.data.frame(summ$pcor_results))
})

test_that("summary.var_estimate beta_results is list", {
  fit <- var_estimate(test_ts_data, iter = 50, progress = FALSE)
  summ <- summary(fit)

  expect_true(is.list(summ$beta_results))
  expect_equal(length(summ$beta_results), p)
})

test_that("summary.var_estimate respects cred parameter", {
  fit <- var_estimate(test_ts_data, iter = 50, progress = FALSE)

  summ_95 <- summary(fit, cred = 0.95)
  summ_90 <- summary(fit, cred = 0.90)

  expect_true(is.data.frame(summ_95$pcor_results))
  expect_true(is.data.frame(summ_90$pcor_results))
})

test_that("summary.var_estimate has correct column names", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 50, progress = FALSE)
  summ <- summary(fit)

  expected_cols <- c("Relation", "Post.mean", "Post.sd", "Cred.lb", "Cred.ub")
  expect_equal(colnames(summ$pcor_results), expected_cols)
})

test_that("print.summary.var_estimate works for all params", {
  fit <- var_estimate(test_ts_data, iter = 50, progress = FALSE)
  summ <- summary(fit)

  output <- capture.output(print(summ, param = "all"))

  expect_true(length(output) > 0)
  expect_true(any(grepl("Partial Correlations", output)))
  expect_true(any(grepl("Coefficients", output)))
})

test_that("print.summary.var_estimate works for pcor only", {
  fit <- var_estimate(test_ts_data, iter = 50, progress = FALSE)
  summ <- summary(fit)

  output <- capture.output(print(summ, param = "pcor"))

  expect_true(length(output) > 0)
  expect_true(any(grepl("Partial Correlations", output)))
})

test_that("print.summary.var_estimate works for beta only", {
  fit <- var_estimate(test_ts_data, iter = 50, progress = FALSE)
  summ <- summary(fit)

  output <- capture.output(print(summ, param = "beta"))

  expect_true(length(output) > 0)
  expect_true(any(grepl("Coefficients", output)))
})

test_that("plot.summary.var_estimate works", {
  fit <- var_estimate(test_ts_data, iter = 50, progress = FALSE)
  summ <- summary(fit)

  plt <- plot(summ)

  expect_true(is.list(plt))
  expect_true("pcor_plt" %in% names(plt))
  expect_true("beta_plt" %in% names(plt))
})

test_that("plot.summary.var_estimate pcor only", {
  fit <- var_estimate(test_ts_data, iter = 50, progress = FALSE)
  summ <- summary(fit)

  plt <- plot(summ, param = "pcor")

  expect_true(inherits(plt$pcor_plt, "ggplot"))
  expect_null(plt$beta_plt)
})

test_that("plot.summary.var_estimate beta only", {
  fit <- var_estimate(test_ts_data, iter = 50, progress = FALSE)
  summ <- summary(fit)

  plt <- plot(summ, param = "beta")

  expect_null(plt$pcor_plt)
  expect_true(is.list(plt$beta_plt))
})

test_that("plot.summary.var_estimate respects order parameter", {
  fit <- var_estimate(test_ts_data, iter = 50, progress = FALSE)
  summ <- summary(fit)

  plt <- plot(summ, order = FALSE)

  expect_true(is.list(plt))
})

test_that("plot.summary.var_estimate respects color parameter", {
  fit <- var_estimate(test_ts_data, iter = 50, progress = FALSE)
  summ <- summary(fit)

  plt <- plot(summ, color = "red")

  expect_true(is.list(plt))
})

test_that("plot.summary.var_estimate respects size parameter", {
  fit <- var_estimate(test_ts_data, iter = 50, progress = FALSE)
  summ <- summary(fit)

  plt <- plot(summ, size = 3)

  expect_true(is.list(plt))
})

test_that("plot.summary.var_estimate respects width parameter", {
  fit <- var_estimate(test_ts_data, iter = 50, progress = FALSE)
  summ <- summary(fit)

  plt <- plot(summ, width = 0.5)

  expect_true(is.list(plt))
})

test_that("var_estimate stores correct n and p", {
  fit <- var_estimate(test_ts_data, iter = 50, progress = FALSE)

  expect_equal(fit$p, p)
  expect_true(fit$n > 0)
})

test_that("var_estimate stores Y and X matrices", {
  fit <- var_estimate(test_ts_data, iter = 50, progress = FALSE)

  expect_true("Y" %in% names(fit))
  expect_true("X" %in% names(fit))
  expect_true(is.matrix(fit$Y))
  expect_true(is.matrix(fit$X))
})
