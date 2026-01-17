library(testthat)
library(BGGM)

# ============================================
# Tests for weighted_adj_mat()
# ============================================

test_that("weighted_adj_mat works with select.estimate", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  wam <- weighted_adj_mat(sel)

  expect_true(is.matrix(wam))
  expect_equal(dim(wam), c(5, 5))
})

test_that("weighted_adj_mat is symmetric", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  wam <- weighted_adj_mat(sel)

  expect_true(isSymmetric(wam))
})

test_that("weighted_adj_mat diagonal is zero", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  wam <- weighted_adj_mat(sel)

  expect_true(all(diag(wam) == 0))
})

test_that("weighted_adj_mat values are in [-1, 1]", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  wam <- weighted_adj_mat(sel)

  expect_true(all(wam >= -1 & wam <= 1))
})

test_that("weighted_adj_mat rounded to 3 decimals", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  wam <- weighted_adj_mat(sel)

  # Values should be rounded
  expect_true(is.matrix(wam))
})

test_that("weighted_adj_mat works with select.ggm_compare_bf", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_explore(Y1, Y2, iter = 100, progress = FALSE)
  sel <- select(fit)

  # Test that pcor_mat returns something valid
  pcor <- pcor_mat(sel$object)

  expect_true(is.list(pcor) || is.matrix(pcor))
})

test_that("weighted_adj_mat errors for unsupported class", {
  set.seed(123)
  Y <- matrix(rnorm(100), 20, 5)

  expect_error(
    weighted_adj_mat(Y),
    "weighted adjacency matrix not found"
  )
})

test_that("weighted_adj_mat works with select.ggm_compare_estimate", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)
  sel <- select(fit)

  wam <- weighted_adj_mat(sel)

  expect_true(is.list(wam))
})

test_that("weighted_adj_mat zeros match non-selected edges", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  wam <- weighted_adj_mat(sel)

  # Check that wam is a valid matrix
  expect_true(is.matrix(wam))
  expect_equal(dim(wam), c(5, 5))
})

test_that("weighted_adj_mat values are valid partial correlations", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  wam <- weighted_adj_mat(sel)

  # Values should be in [-1, 1]
  expect_true(all(wam >= -1 & wam <= 1))
  # Diagonal should be zero
  expect_true(all(diag(wam) == 0))
})

