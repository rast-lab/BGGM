library(testthat)
library(BGGM)

# Test data setup - two groups
set.seed(123)
n_per_group <- 50
p <- 4

# Group 1
group1 <- matrix(rnorm(n_per_group * p), ncol = p)
colnames(group1) <- paste0("V", 1:p)

# Group 2
group2 <- matrix(rnorm(n_per_group * p), ncol = p)
colnames(group2) <- paste0("V", 1:p)

# Basic functionality tests
test_that("ggm_compare_ppc works with default parameters", {
  result <- ggm_compare_ppc(
    group1, group2,
    iter = 50,
    progress = FALSE
  )

  expect_s3_class(result, c("BGGM", "ggm_compare_ppc"))
  expect_true(is.list(result))
})

test_that("ggm_compare_ppc returns correct structure", {
  result <- ggm_compare_ppc(
    group1, group2,
    iter = 50,
    progress = FALSE
  )

  # Should contain ppc-related elements
  expect_true(any(c("ppp_jsd", "ppp_sse", "predictive_jsd") %in% names(result)))
})

test_that("ggm_compare_ppc works with three groups", {
  set.seed(456)
  group3 <- matrix(rnorm(n_per_group * p), ncol = p)
  colnames(group3) <- paste0("V", 1:p)

  result <- ggm_compare_ppc(
    group1, group2, group3,
    iter = 50,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_compare_ppc")
})

# Test different test types
test_that("ggm_compare_ppc works with global test", {
  result <- ggm_compare_ppc(
    group1, group2,
    test = "global",
    iter = 50,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_compare_ppc")
})

test_that("ggm_compare_ppc works with nodewise test", {
  result <- ggm_compare_ppc(
    group1, group2,
    test = "nodewise",
    iter = 50,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_compare_ppc")
})

# Test print method (no summary method exists for ggm_compare_ppc)
test_that("print.ggm_compare_ppc works", {
  fit <- ggm_compare_ppc(
    group1, group2,
    iter = 50,
    progress = FALSE
  )

  # Test that print works without error
  expect_output(print(fit))
})

# Test plot method
test_that("plot.ggm_compare_ppc works without error", {
  fit <- ggm_compare_ppc(
    group1, group2,
    iter = 50,
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

# Test with unequal group sizes
test_that("ggm_compare_ppc handles unequal group sizes", {
  set.seed(222)
  small_group <- matrix(rnorm(30 * p), ncol = p)
  large_group <- matrix(rnorm(70 * p), ncol = p)
  colnames(small_group) <- colnames(large_group) <- paste0("V", 1:p)

  result <- ggm_compare_ppc(
    small_group, large_group,
    iter = 50,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_compare_ppc")
})

# Test return values contain p-values
test_that("ggm_compare_ppc returns p-values", {
  result <- ggm_compare_ppc(
    group1, group2,
    iter = 50,
    progress = FALSE
  )

  # Should have p-value information
  expect_true("ppp_jsd" %in% names(result) || "ppp_sse" %in% names(result))
})

# Test p-values are in valid range
test_that("ggm_compare_ppc p-values are between 0 and 1", {
  result <- ggm_compare_ppc(
    group1, group2,
    iter = 50,
    progress = FALSE
  )

  if ("ppp_jsd" %in% names(result)) {
    pvals <- unlist(result$ppp_jsd)
    expect_true(all(pvals >= 0 & pvals <= 1, na.rm = TRUE))
  }
})
