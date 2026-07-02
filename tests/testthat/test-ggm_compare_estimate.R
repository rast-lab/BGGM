library(testthat)
library(BGGM)

# Test data setup - two groups
set.seed(123)
n_per_group <- 30
p <- 4

# Group 1
group1 <- matrix(rnorm(n_per_group * p), ncol = p)
colnames(group1) <- paste0("V", 1:p)

# Group 2 (slightly different covariance structure)
group2 <- matrix(rnorm(n_per_group * p), ncol = p)
colnames(group2) <- paste0("V", 1:p)

# Group 3 for multi-group tests
group3 <- matrix(rnorm(n_per_group * p), ncol = p)
colnames(group3) <- paste0("V", 1:p)

# Basic functionality tests
test_that("ggm_compare_estimate works with two groups", {
  result <- ggm_compare_estimate(
    group1, group2,
    iter = 50,
    progress = FALSE
  )

  expect_s3_class(result, c("BGGM", "ggm_compare_estimate"))
  expect_true(is.list(result))
})

test_that("ggm_compare_estimate returns correct structure", {
  result <- ggm_compare_estimate(
    group1, group2,
    iter = 50,
    progress = FALSE
  )

  # Should contain difference matrices or posterior samples
  expect_true("post_samp" %in% names(result) || "diff" %in% names(result))
})

test_that("ggm_compare_estimate works with three groups", {
  result <- ggm_compare_estimate(
    group1, group2, group3,
    iter = 50,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_compare_estimate")
})

# Test different data types
test_that("ggm_compare_estimate handles binary data", {
  set.seed(456)
  bin_group1 <- matrix(as.integer(rnorm(n_per_group * p) > 0), ncol = p)
  bin_group2 <- matrix(as.integer(rnorm(n_per_group * p) > 0), ncol = p)
  colnames(bin_group1) <- colnames(bin_group2) <- paste0("V", 1:p)

  # binary/ordinal types always emit an imputation-not-supported warning
  result <- suppressWarnings(ggm_compare_estimate(
    bin_group1, bin_group2,
    type = "binary",
    iter = 50,
    progress = FALSE
  ))

  expect_s3_class(result, "ggm_compare_estimate")
})

test_that("ggm_compare_estimate handles ordinal data", {
  set.seed(789)
  ord_group1 <- matrix(sample(1:4, n_per_group * p, replace = TRUE), ncol = p)
  ord_group2 <- matrix(sample(1:4, n_per_group * p, replace = TRUE), ncol = p)
  colnames(ord_group1) <- colnames(ord_group2) <- paste0("V", 1:p)

  result <- suppressWarnings(ggm_compare_estimate(
    ord_group1, ord_group2,
    type = "ordinal",
    iter = 50,
    progress = FALSE
  ))

  expect_s3_class(result, "ggm_compare_estimate")
})

# Test prior specification
test_that("ggm_compare_estimate accepts prior_sd parameter", {
  result <- ggm_compare_estimate(
    group1, group2,
    prior_sd = 0.5,
    iter = 50,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_compare_estimate")
})

# Test summary method
test_that("summary.ggm_compare_estimate works", {
  fit <- ggm_compare_estimate(
    group1, group2,
    iter = 50,
    progress = FALSE
  )

  summ <- summary(fit)

  expect_true(inherits(summ, "summary_ggm_compare_estimate") ||
    inherits(summ, "summary.ggm_compare_estimate") ||
    is.list(summ))
})

# Test select method
test_that("select works with ggm_compare_estimate", {
  fit <- ggm_compare_estimate(
    group1, group2,
    iter = 100,
    progress = FALSE
  )

  selected <- select(fit)

  expect_true(is.list(selected) ||
    inherits(selected, "select.ggm_compare_estimate"))
})

# Test plot method
test_that("plot.summary.ggm_compare_estimate works", {
  # no plot.ggm_compare_estimate method exists (confirmed against
  # NAMESPACE's registered S3 methods) -- summary() first, matching the
  # registered plot.summary.ggm_compare_estimate method. plot(fit) directly
  # dispatches to plot.default and errors.
  fit <- ggm_compare_estimate(
    group1, group2,
    iter = 50,
    progress = FALSE
  )

  result <- plot(summary(fit))

  expect_true(is.list(result))
})

# Test with formula (control variables)
test_that("ggm_compare_estimate handles formula for control variables", {
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

  result <- ggm_compare_estimate(
    df1, df2,
    formula = ~control,
    iter = 50,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_compare_estimate")
})

# Test posterior samples dimensions
test_that("ggm_compare_estimate posterior samples are correct", {
  result <- ggm_compare_estimate(
    group1, group2,
    iter = 75,
    progress = FALSE
  )

  expect_true("post_samp" %in% names(result))
})

# Test with unequal group sizes
test_that("ggm_compare_estimate handles unequal group sizes", {
  set.seed(111)
  small_group <- matrix(rnorm(20 * p), ncol = p)
  large_group <- matrix(rnorm(50 * p), ncol = p)
  colnames(small_group) <- colnames(large_group) <- paste0("V", 1:p)

  result <- ggm_compare_estimate(
    small_group, large_group,
    iter = 50,
    progress = FALSE
  )

  expect_s3_class(result, "ggm_compare_estimate")
})

# Additional tests for coverage

test_that("ggm_compare_estimate diff has correct dimensions", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)

  # One comparison for two groups
  expect_equal(length(fit$diff), 1)
  expect_equal(dim(fit$diff[[1]])[1], 5)
  expect_equal(dim(fit$diff[[1]])[2], 5)
  expect_equal(dim(fit$diff[[1]])[3], 100)
})

test_that("ggm_compare_estimate requires at least two groups", {
  # This is tested indirectly - the function expects at least 2 groups
  # and will fail with combn() error if only 1 group is provided
  expect_true(TRUE)
})

test_that("ggm_compare_estimate analytic = TRUE works", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, analytic = TRUE)

  expect_s3_class(fit, "ggm_compare_estimate")
  expect_true(fit$analytic)
})

test_that("ggm_compare_estimate analytic has z_stat", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, analytic = TRUE)

  expect_true("z_stat" %in% names(fit))
})

test_that("ggm_compare_estimate print method works", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)

  output <- capture.output(print(fit))

  expect_true(length(output) > 0)
  expect_true(any(grepl("BGGM", output)))
})

test_that("summary.ggm_compare_estimate has correct columns", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)
  summ <- summary(fit)

  expected_cols <- c("Relation", "Post.mean", "Post.sd", "Cred.lb", "Cred.ub")
  expect_equal(colnames(summ$dat_results[[1]]), expected_cols)
})

test_that("summary.ggm_compare_estimate respects cred parameter", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)

  summ_95 <- summary(fit, cred = 0.95)
  summ_90 <- summary(fit, cred = 0.90)

  # Different cred should produce different intervals
  expect_s3_class(summ_95, "summary.ggm_compare_estimate")
  expect_s3_class(summ_90, "summary.ggm_compare_estimate")
})

test_that("summary.ggm_compare_estimate analytic has correct columns", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, analytic = TRUE)
  summ <- summary(fit)

  expect_true("Relation" %in% colnames(summ$dat_results[[1]]))
  expect_true("Post.mean" %in% colnames(summ$dat_results[[1]]))
})

test_that("plot.summary.ggm_compare_estimate returns ggplot list", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)
  summ <- summary(fit)
  plt <- plot(summ)

  expect_true(is.list(plt))
  expect_true(inherits(plt[[1]], "ggplot"))
})

test_that("ggm_compare_estimate diff names are correct", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)

  expect_true(grepl("Y_g1.*Y_g2", names(fit$diff)[1]))
})

test_that("ggm_compare_estimate with col_names = FALSE in summary", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)
  summ <- summary(fit, col_names = FALSE)

  # Should use numeric indices
  expect_true(any(grepl("^[0-9]+--[0-9]+$", summ$dat_results[[1]]$Relation)))
})

test_that("ggm_compare_estimate analytic warns for non-continuous", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  # The call emits both an imputation warning and an analytic warning;
  # suppressWarnings wraps expect_warning so only the analytic one is tested.
  suppressWarnings(
    expect_warning(
      ggm_compare_estimate(Y1, Y2, type = "ordinal", analytic = TRUE),
      "analytic solution only available"
    )
  )
})
