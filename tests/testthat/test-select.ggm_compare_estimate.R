library(testthat)
library(BGGM)

# ============================================
# Tests for select.ggm_compare_estimate()
# ============================================

test_that("select.ggm_compare_estimate returns correct class", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)
  sel <- select(fit)

  expect_s3_class(sel, "BGGM")
  expect_s3_class(sel, "select.ggm_compare_estimate")
})

test_that("select.ggm_compare_estimate returns expected components", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)
  sel <- select(fit)

  expect_true("mean_diff" %in% names(sel))
  expect_true("pcor_adj" %in% names(sel))
  expect_true("adj" %in% names(sel))
})

test_that("select.ggm_compare_estimate adj is list", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)
  sel <- select(fit)

  expect_true(is.list(sel$adj))
  expect_equal(length(sel$adj), 1)  # One comparison
})

test_that("select.ggm_compare_estimate adj is binary", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)
  sel <- select(fit)

  expect_true(all(sel$adj[[1]] %in% c(0, 1)))
})

test_that("select.ggm_compare_estimate pcor_adj correct dimensions", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)
  sel <- select(fit)

  expect_equal(dim(sel$pcor_adj[[1]]), c(5, 5))
})

test_that("select.ggm_compare_estimate respects cred parameter", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)

  sel_95 <- select(fit, cred = 0.95)
  sel_99 <- select(fit, cred = 0.99)

  # Higher cred should have fewer or equal selected edges
  expect_true(sum(sel_99$adj[[1]]) <= sum(sel_95$adj[[1]]))
})

test_that("select.ggm_compare_estimate with three groups", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:30, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:30, ]
  Y3 <- BGGM::bfi[61:90, 1:5]

  fit <- ggm_compare_estimate(Y1, Y2, Y3, iter = 100, progress = FALSE)
  sel <- select(fit)

  # Should have 3 comparisons: 1-2, 1-3, 2-3
  expect_equal(length(sel$adj), 3)
})

test_that("select.ggm_compare_estimate print method works", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)
  sel <- select(fit)

  output <- capture.output(print(sel))

  expect_true(length(output) > 0)
  expect_true(any(grepl("BGGM", output)))
  expect_true(any(grepl("Selected", output)))
})

test_that("select.ggm_compare_estimate with analytic = TRUE", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, analytic = TRUE)
  sel <- select(fit)

  expect_s3_class(sel, "select.ggm_compare_estimate")
  expect_true("adj" %in% names(sel))
})

test_that("select.ggm_compare_estimate analytic adj is binary", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, analytic = TRUE)
  sel <- select(fit)

  expect_true(all(sel$adj[[1]] %in% c(0, 1)))
})

test_that("select.ggm_compare_estimate analytic respects cred", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, analytic = TRUE)

  sel_95 <- select(fit, cred = 0.95)
  sel_99 <- select(fit, cred = 0.99)

  # Higher cred should have fewer or equal selected edges
  expect_true(sum(sel_99$adj[[1]]) <= sum(sel_95$adj[[1]]))
})

test_that("select.ggm_compare_estimate stores cred value", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)
  sel <- select(fit, cred = 0.90)

  expect_equal(sel$cred, 0.90)
})

test_that("select.ggm_compare_estimate mean_diff is list", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)
  sel <- select(fit)

  expect_true(is.list(sel$mean_diff))
})

test_that("select.ggm_compare_estimate pcor_adj matches adj", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)
  sel <- select(fit)

  # Where adj is 0, pcor_adj should also be 0
  zero_positions <- which(sel$adj[[1]] == 0)
  expect_true(all(sel$pcor_adj[[1]][zero_positions] == 0))
})

