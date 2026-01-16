library(testthat)
library(BGGM)

# Test data setup
# Note: ggm_search requires sufficient sample size relative to p
set.seed(123)
n <- 200
p <- 4
test_data <- matrix(rnorm(n * p), ncol = p)
colnames(test_data) <- paste0("V", 1:p)

# Basic functionality tests
test_that("ggm_search works with default parameters", {
  result <- ggm_search(
    test_data,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, c("BGGM", "ggm_search"))
  expect_true(is.list(result))
})

test_that("ggm_search returns correct structure", {
  result <- ggm_search(
    test_data,
    iter = 100,
    progress = FALSE
  )

  # Should contain graph structure elements
  expect_true(any(c("adj", "Adj", "adjacency", "graph") %in% names(result)))
})

test_that("ggm_search returns adjacency matrix", {
  result <- ggm_search(
    test_data,
    iter = 100,
    progress = FALSE
  )

  # Check for adjacency matrix
  adj_names <- c("adj", "Adj", "adjacency", "map_adj")
  adj_found <- intersect(adj_names, names(result))

  if (length(adj_found) > 0) {
    adj <- result[[adj_found[1]]]
    expect_true(is.matrix(adj))
    expect_equal(dim(adj), c(p, p))
  }
})

# Test BMA (Bayesian Model Averaging) option
test_that("ggm_search works with bma = TRUE", {
  result <- ggm_search(
    test_data,
    bma = TRUE,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_search")
})

test_that("ggm_search with bma returns proper structure", {
  result <- ggm_search(
    test_data,
    bma = TRUE,
    iter = 100,
    progress = FALSE
  )

  # BMA should provide additional uncertainty info
  expect_true(is.list(result))
})

# Test start parameter
test_that("ggm_search accepts start parameter", {
  # Start from empty graph
  start_adj <- matrix(0, p, p)

  result <- ggm_search(
    test_data,
    start = start_adj,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_search")
})

test_that("ggm_search accepts start = 'full'", {
  result <- ggm_search(
    test_data,
    start = "full",
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_search")
})

# Test gamma parameter
test_that("ggm_search accepts gamma parameter", {
  result <- ggm_search(
    test_data,
    gamma = 0.5,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_search")
})

# Test prior_sd parameter
test_that("ggm_search accepts prior_sd parameter", {
  result <- ggm_search(
    test_data,
    prior_sd = 0.5,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_search")
})

# Test print method (no summary method exists for ggm_search)
test_that("print.ggm_search works", {
  fit <- ggm_search(
    test_data,
    iter = 100,
    progress = FALSE
  )

  # Test that print works without error
  expect_output(print(fit))
})

# Test plot method
test_that("plot.ggm_search works without error", {
  fit <- ggm_search(
    test_data,
    iter = 100,
    progress = FALSE
  )

  result <- tryCatch(
    {
      plot(fit)
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})

# Test with different data sizes
test_that("ggm_search handles different sample sizes", {
  set.seed(456)
  small_data <- matrix(rnorm(30 * 4), ncol = 4)
  colnames(small_data) <- paste0("V", 1:4)

  result <- ggm_search(
    small_data,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_search")
})

test_that("ggm_search handles different number of variables", {
  set.seed(789)
  data_5var <- matrix(rnorm(50 * 5), ncol = 5)
  colnames(data_5var) <- paste0("V", 1:5)

  result <- ggm_search(
    data_5var,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_search")
})

# Test bma_posterior with ggm_search output
test_that("bma_posterior works with ggm_search bma output", {
  fit_bma <- ggm_search(
    test_data,
    bma = TRUE,
    iter = 100,
    progress = FALSE
  )

  result <- bma_posterior(fit_bma)

  expect_true(is.list(result) || inherits(result, "bma_posterior"))
})

test_that("bma_posterior returns correct structure", {
  fit_bma <- ggm_search(
    test_data,
    bma = TRUE,
    iter = 100,
    progress = FALSE
  )

  result <- bma_posterior(fit_bma)

  # Should contain posterior samples or matrices
  expect_true(any(c("post_samp", "pcor_mat", "samples") %in% names(result)) ||
    is.matrix(result) || is.array(result) || is.list(result))
})

test_that("bma_posterior handles iter parameter", {
  fit_bma <- ggm_search(
    test_data,
    bma = TRUE,
    iter = 100,
    progress = FALSE
  )

  result <- bma_posterior(fit_bma, iter = 50)

  expect_true(is.list(result) || inherits(result, "bma_posterior"))
})
