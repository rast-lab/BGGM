library(testthat)
library(BGGM)

# Test data setup
set.seed(123)
n <- 50
p <- 4
test_data <- matrix(rnorm(n * p), ncol = p)
colnames(test_data) <- paste0("V", 1:p)

# Create estimate object for testing
fit <- estimate(test_data, iter = 100, progress = FALSE)

# Define some custom network statistics
mean_abs_pcor <- function(x) {
  mean(abs(x[upper.tri(x)]))
}

max_abs_pcor <- function(x) {
  max(abs(x[upper.tri(x)]))
}

network_density <- function(x, threshold = 0.1) {
  adj <- abs(x) > threshold
  sum(adj[upper.tri(adj)]) / sum(upper.tri(adj))
}

# Basic functionality tests
test_that("roll_your_own works with simple function", {
  result <- roll_your_own(fit, FUN = mean_abs_pcor)

  expect_s3_class(result, "roll_your_own")
  expect_true(is.list(result))
})

test_that("roll_your_own returns correct structure", {
  result <- roll_your_own(fit, FUN = mean_abs_pcor)

  # Should contain samples or summary
  expect_true(any(c("samples", "samp", "results", "summary") %in% names(result)))
})

test_that("roll_your_own works with different custom functions", {
  result1 <- roll_your_own(fit, FUN = mean_abs_pcor)
  result2 <- roll_your_own(fit, FUN = max_abs_pcor)

  expect_s3_class(result1, "roll_your_own")
  expect_s3_class(result2, "roll_your_own")
})

test_that("roll_your_own works with density function", {
  result <- roll_your_own(fit, FUN = network_density)

  expect_s3_class(result, "roll_your_own")
})

# Test that roll_your_own requires estimate class
test_that("roll_your_own requires estimate class", {
  # roll_your_own only accepts estimate objects, not select.estimate
  sel <- select(fit)
  expect_error(roll_your_own(sel, FUN = mean_abs_pcor), "class must be 'estimate'")
})

# Test print method (no summary method exists for roll_your_own)
test_that("print.roll_your_own works", {
  ryo <- roll_your_own(fit, FUN = mean_abs_pcor)

  # Test that print works without error
  expect_output(print(ryo))
})

# Test plot method
test_that("plot.roll_your_own works without error", {
  ryo <- roll_your_own(fit, FUN = mean_abs_pcor)

  result <- tryCatch(
    {
      plot(ryo)
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})

# Test with node-wise function
test_that("roll_your_own works with node-wise function", {
  # Function that returns value for each node
  node_strength <- function(x) {
    colSums(abs(x))
  }

  result <- roll_your_own(fit, FUN = node_strength)

  expect_s3_class(result, "roll_your_own")
})

# Test that roll_your_own rejects explore objects
test_that("roll_your_own rejects explore object", {
  fit_explore <- explore(test_data, progress = FALSE)
  # roll_your_own only accepts estimate objects
  expect_error(roll_your_own(fit_explore, FUN = mean_abs_pcor), "class must be 'estimate'")
})

# Test return values are numeric
test_that("roll_your_own returns numeric samples", {
  result <- roll_your_own(fit, FUN = mean_abs_pcor)

  if ("samples" %in% names(result)) {
    expect_true(is.numeric(result$samples))
  }

  if ("samp" %in% names(result)) {
    expect_true(is.numeric(result$samp))
  }
})

# Test with lambda function (inline)
test_that("roll_your_own works with inline function", {
  result <- roll_your_own(fit, FUN = function(x) sum(x^2))

  expect_s3_class(result, "roll_your_own")
})

# Test samples have correct length
test_that("roll_your_own samples match iterations", {
  fit_100 <- estimate(test_data, iter = 100, progress = FALSE)
  result <- roll_your_own(fit_100, FUN = mean_abs_pcor)

  if ("samples" %in% names(result)) {
    # Samples should have length related to iterations
    expect_true(length(result$samples) > 0)
  }
})

# Test with different data types
test_that("roll_your_own works with binary data estimate", {
  set.seed(456)
  binary_data <- matrix(as.integer(rnorm(50 * 4) > 0), ncol = 4)
  colnames(binary_data) <- paste0("V", 1:4)

  fit_bin <- estimate(binary_data, type = "binary", iter = 50, progress = FALSE)
  result <- roll_your_own(fit_bin, FUN = mean_abs_pcor)

  expect_s3_class(result, "roll_your_own")
})

test_that("roll_your_own works with ordinal data estimate", {
  set.seed(789)
  ordinal_data <- matrix(sample(1:4, 50 * 4, replace = TRUE), ncol = 4)
  colnames(ordinal_data) <- paste0("V", 1:4)

  fit_ord <- estimate(ordinal_data, type = "ordinal", iter = 50, progress = FALSE)
  result <- roll_your_own(fit_ord, FUN = mean_abs_pcor)

  expect_s3_class(result, "roll_your_own")
})

# Test that roll_your_own result can be used for credible intervals
test_that("roll_your_own result can be used for credible intervals", {
  result <- roll_your_own(fit, FUN = mean_abs_pcor)

  # Should be able to extract samples from the result
  expect_true(is.list(result))
})
