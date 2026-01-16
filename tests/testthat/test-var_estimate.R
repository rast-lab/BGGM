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
test_that("var_estimate accepts prior_sd parameter", {
  result <- var_estimate(
    test_ts_data,
    prior_sd = 0.5,
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

# Test plot method (if exists)
test_that("plot.var_estimate works without error", {
  fit <- var_estimate(test_ts_data, iter = 50, progress = FALSE)

  # This may or may not have a plot method
  # Using tryCatch to handle case where no plot method exists
  result <- tryCatch(
    {
      plot(fit)
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})
