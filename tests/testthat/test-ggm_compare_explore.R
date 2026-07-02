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

# Group 3 for multi-group tests
group3 <- matrix(rnorm(n_per_group * p), ncol = p)
colnames(group3) <- paste0("V", 1:p)

# Basic functionality tests
test_that("ggm_compare_explore works with two groups", {
  result <- ggm_compare_explore(
    group1, group2,
    progress = FALSE
  )

  expect_s3_class(result, c("BGGM", "ggm_compare_explore"))
  expect_true(is.list(result))
})

test_that("ggm_compare_explore returns correct structure", {
  result <- ggm_compare_explore(
    group1, group2,
    progress = FALSE
  )

  # Should contain the Bayes factor and pairwise partial-correlation
  # difference fields
  expect_true(all(c("BF_01", "pcor_diff", "info") %in% names(result)))
})

test_that("ggm_compare_explore works with three groups", {
  result <- ggm_compare_explore(
    group1, group2, group3,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_compare_explore")
})

# Test different data types
test_that("ggm_compare_explore handles binary data", {
  set.seed(456)
  bin_group1 <- matrix(as.integer(rnorm(n_per_group * p) > 0), ncol = p)
  bin_group2 <- matrix(as.integer(rnorm(n_per_group * p) > 0), ncol = p)
  colnames(bin_group1) <- colnames(bin_group2) <- paste0("V", 1:p)

  result <- ggm_compare_explore(
    bin_group1, bin_group2,
    type = "binary",
    iter = 50,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_compare_explore")
})

test_that("ggm_compare_explore handles ordinal data", {
  set.seed(789)
  ord_group1 <- matrix(sample(1:4, n_per_group * p, replace = TRUE), ncol = p)
  ord_group2 <- matrix(sample(1:4, n_per_group * p, replace = TRUE), ncol = p)
  colnames(ord_group1) <- colnames(ord_group2) <- paste0("V", 1:p)

  result <- ggm_compare_explore(
    ord_group1, ord_group2,
    type = "ordinal",
    iter = 50,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_compare_explore")
})

# Test prior specification
test_that("ggm_compare_explore accepts prior_sd parameter", {
  result <- ggm_compare_explore(
    group1, group2,
    prior_sd = 0.5,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_compare_explore")
})

# Test summary method
test_that("summary.ggm_compare_explore works", {
  fit <- ggm_compare_explore(
    group1, group2,
    progress = FALSE
  )

  summ <- summary(fit)

  expect_true(inherits(summ, "summary_ggm_compare_explore") ||
    inherits(summ, "summary.ggm_compare_explore") ||
    is.list(summ))
})

# Test select method
test_that("select works with ggm_compare_explore", {
  fit <- ggm_compare_explore(
    group1, group2,
    progress = FALSE
  )

  selected <- select(fit)

  expect_true(is.list(selected) ||
    inherits(selected, "select.ggm_compare_explore"))
})

# Test plot method
test_that("plot.summary.ggm_compare_explore works", {
  # no plot.ggm_compare_explore method exists (confirmed against
  # NAMESPACE's registered S3 methods) -- summary() first, matching the
  # registered plot.summary.ggm_compare_explore method. plot(fit) directly
  # dispatches to plot.default and errors.
  fit <- ggm_compare_explore(
    group1, group2,
    progress = FALSE
  )

  result <- plot(summary(fit))

  expect_s3_class(result, "ggplot")
})

# Test with formula (control variables)
test_that("ggm_compare_explore handles formula for control variables", {
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

  result <- ggm_compare_explore(
    df1, df2,
    formula = ~control,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_compare_explore")
})

# Test with unequal group sizes
test_that("ggm_compare_explore handles unequal group sizes", {
  set.seed(111)
  small_group <- matrix(rnorm(20 * p), ncol = p)
  large_group <- matrix(rnorm(50 * p), ncol = p)
  colnames(small_group) <- colnames(large_group) <- paste0("V", 1:p)

  result <- ggm_compare_explore(
    small_group, large_group,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_compare_explore")
})

# Test that select() BF_cut threshold works on ggm_compare_explore results
test_that("select.ggm_compare_explore respects BF_cut threshold", {
  set.seed(12345)
  Y1 <- BGGM::bfi[1:200, 1:4]
  Y2 <- BGGM::bfi[201:400, 1:4]

  result <- ggm_compare_explore(Y1, Y2, progress = FALSE)
  sel_3  <- select(result, BF_cut = 3)
  sel_10 <- select(result, BF_cut = 10)

  # Stricter BF threshold should select fewer or equal edges
  edges_3  <- sum(sel_3$adj_10[upper.tri(sel_3$adj_10)])
  edges_10 <- sum(sel_10$adj_10[upper.tri(sel_10$adj_10)])
  expect_true(edges_10 <= edges_3)
})
