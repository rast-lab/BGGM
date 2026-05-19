library(testthat)
library(BGGM)

# Test data setup - two groups
set.seed(123)
n_per_group <- 30
p <- 4

# Group 1
group1 <- matrix(rnorm(n_per_group * p), ncol = p)
colnames(group1) <- paste0("V", 1:p)

# Group 2
group2 <- matrix(rnorm(n_per_group * p), ncol = p)
colnames(group2) <- paste0("V", 1:p)

# Basic functionality tests
test_that("ggm_compare_confirm works with equality hypothesis", {
  # Hypothesis: edge V1--V2 is equal across groups
  hypothesis <- "g1_V1--V2 = g2_V1--V2"

  result <- ggm_compare_confirm(
    group1, group2,
    hypothesis = hypothesis,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, c("BGGM", "ggm_compare_confirm"))
  expect_true(is.list(result))
})

test_that("ggm_compare_confirm returns correct structure", {
  hypothesis <- "g1_V1--V2 = g2_V1--V2"

  result <- ggm_compare_confirm(
    group1, group2,
    hypothesis = hypothesis,
    iter = 100,
    progress = FALSE
  )

  expect_true("hypothesis" %in% names(result))
})

test_that("ggm_compare_confirm works with inequality hypothesis (greater)", {
  # Hypothesis: edge in group 1 is greater than in group 2
  hypothesis <- "g1_V1--V2 > g2_V1--V2"

  result <- ggm_compare_confirm(
    group1, group2,
    hypothesis = hypothesis,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_compare_confirm")
})

test_that("ggm_compare_confirm works with inequality hypothesis (less)", {
  # Hypothesis: edge in group 1 is less than in group 2
  hypothesis <- "g1_V1--V2 < g2_V1--V2"

  result <- ggm_compare_confirm(
    group1, group2,
    hypothesis = hypothesis,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_compare_confirm")
})

test_that("ggm_compare_confirm works with multiple edges hypothesis", {
  # Hypothesis: multiple edges are equal across groups
  hypothesis <- "g1_V1--V2 = g2_V1--V2; g1_V3--V4 = g2_V3--V4"

  result <- ggm_compare_confirm(
    group1, group2,
    hypothesis = hypothesis,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_compare_confirm")
})

# Test different data types
test_that("ggm_compare_confirm handles binary data", {
  set.seed(456)
  bin_group1 <- matrix(as.integer(rnorm(n_per_group * p) > 0), ncol = p)
  bin_group2 <- matrix(as.integer(rnorm(n_per_group * p) > 0), ncol = p)
  colnames(bin_group1) <- colnames(bin_group2) <- paste0("V", 1:p)

  hypothesis <- "g1_V1--V2 = g2_V1--V2"

  # binary/ordinal types always emit an imputation-not-supported warning
  result <- suppressWarnings(ggm_compare_confirm(
    bin_group1, bin_group2,
    hypothesis = hypothesis,
    type = "binary",
    iter = 50,
    progress = FALSE
  ))

  expect_s3_class(result, "ggm_compare_confirm")
})

test_that("ggm_compare_confirm handles ordinal data", {
  set.seed(789)
  ord_group1 <- matrix(sample(1:4, n_per_group * p, replace = TRUE), ncol = p)
  ord_group2 <- matrix(sample(1:4, n_per_group * p, replace = TRUE), ncol = p)
  colnames(ord_group1) <- colnames(ord_group2) <- paste0("V", 1:p)

  hypothesis <- "g1_V1--V2 = g2_V1--V2"

  result <- suppressWarnings(ggm_compare_confirm(
    ord_group1, ord_group2,
    hypothesis = hypothesis,
    type = "ordinal",
    iter = 50,
    progress = FALSE
  ))

  expect_s3_class(result, "ggm_compare_confirm")
})

# Test prior specification
test_that("ggm_compare_confirm accepts prior_sd parameter", {
  hypothesis <- "g1_V1--V2 = g2_V1--V2"

  result <- ggm_compare_confirm(
    group1, group2,
    hypothesis = hypothesis,
    prior_sd = 0.5,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_compare_confirm")
})

# Test print method (no summary method exists for ggm_compare_confirm)
test_that("print.ggm_compare_confirm works", {
  hypothesis <- "g1_V1--V2 = g2_V1--V2"

  fit <- ggm_compare_confirm(
    group1, group2,
    hypothesis = hypothesis,
    iter = 100,
    progress = FALSE
  )

  # Test that print works without error
  expect_output(print(fit))
})

# Test with three groups
test_that("ggm_compare_confirm works with three groups", {
  set.seed(111)
  group3 <- matrix(rnorm(n_per_group * p), ncol = p)
  colnames(group3) <- paste0("V", 1:p)

  # Hypothesis comparing all three groups
  hypothesis <- "g1_V1--V2 = g2_V1--V2 = g3_V1--V2"

  result <- ggm_compare_confirm(
    group1, group2, group3,
    hypothesis = hypothesis,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_compare_confirm")
})

# Test with formula (control variables)
test_that("ggm_compare_confirm handles formula for control variables", {
  set.seed(321)
  # Create data frames with control variable
  df1 <- data.frame(
    V1 = rnorm(30), V2 = rnorm(30), V3 = rnorm(30),
    control = rnorm(30)
  )
  df2 <- data.frame(
    V1 = rnorm(30), V2 = rnorm(30), V3 = rnorm(30),
    control = rnorm(30)
  )

  hypothesis <- "g1_V1--V2 = g2_V1--V2"

  result <- ggm_compare_confirm(
    df1, df2,
    hypothesis = hypothesis,
    formula = ~control,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_compare_confirm")
})
