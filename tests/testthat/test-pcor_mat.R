library(testthat)
library(BGGM)

# ============================================
# Tests for pcor_mat()
# ============================================

test_that("pcor_mat works with estimate object", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)

  pcor <- pcor_mat(fit)

  expect_true(is.matrix(pcor))
  expect_equal(dim(pcor), c(5, 5))
})

test_that("pcor_mat works with explore object", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)

  pcor <- pcor_mat(fit)

  expect_true(is.matrix(pcor))
  expect_equal(dim(pcor), c(5, 5))
})

test_that("pcor_mat is symmetric", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)

  pcor <- pcor_mat(fit)

  expect_true(isSymmetric(pcor))
})

test_that("pcor_mat diagonal is zero", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)

  pcor <- pcor_mat(fit)

  expect_true(all(diag(pcor) == 0))
})

test_that("pcor_mat values are in [-1, 1]", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)

  pcor <- pcor_mat(fit)

  expect_true(all(pcor >= -1 & pcor <= 1))
})

test_that("pcor_mat preserves column names", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)

  pcor <- pcor_mat(fit)

  expect_equal(colnames(pcor), colnames(Y))
  expect_equal(rownames(pcor), colnames(Y))
})

test_that("pcor_mat without column names uses numbers", {
  set.seed(123)
  Y <- as.matrix(BGGM::bfi[1:50, 1:5])
  colnames(Y) <- NULL

  fit <- estimate(Y, iter = 100, progress = FALSE)

  pcor <- pcor_mat(fit)

  expect_equal(colnames(pcor), as.character(1:5))
})

test_that("pcor_mat works with ggm_compare_estimate", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)

  pcor <- pcor_mat(fit)

  expect_true(is.list(pcor))
  expect_true(length(pcor) >= 1)
})

test_that("pcor_mat ggm_compare_estimate difference = TRUE", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)

  pcor <- pcor_mat(fit, difference = TRUE)

  expect_true(is.list(pcor))
})

test_that("pcor_mat with analytic ggm_compare_estimate", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, analytic = TRUE)

  pcor <- pcor_mat(fit)

  expect_true(is.list(pcor))
  expect_equal(length(pcor), 2)
})

test_that("pcor_mat analytic ggm_compare_estimate difference = TRUE", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, analytic = TRUE)

  pcor <- pcor_mat(fit, difference = TRUE)

  expect_true(is.list(pcor))
})

test_that("pcor_mat errors for unsupported class", {
  set.seed(123)
  Y <- matrix(rnorm(100), 20, 5)

  expect_error(
    pcor_mat(Y),
    "partial correlation matrix not found"
  )
})

test_that("pcor_mat works with ggm_compare_explore", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_explore(Y1, Y2, iter = 100, progress = FALSE)

  pcor <- pcor_mat(fit)

  expect_true(is.list(pcor))
  expect_equal(length(pcor), 2)
})

test_that("pcor_mat ggm_compare_explore difference = TRUE", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_explore(Y1, Y2, iter = 100, progress = FALSE)

  pcor <- pcor_mat(fit, difference = TRUE)

  expect_true(is.matrix(pcor))
  expect_equal(dim(pcor), c(5, 5))
})

test_that("pcor_mat ggm_compare_explore difference with more than 2 groups errors", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:30, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:30, ]
  Y3 <- BGGM::bfi[61:90, 1:5]

  fit <- ggm_compare_explore(Y1, Y2, Y3, iter = 100, progress = FALSE)

  expect_error(
    pcor_mat(fit, difference = TRUE),
    "difference only available with two groups"
  )
})

test_that("pcor_mat rounded to 3 decimals", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)

  pcor <- pcor_mat(fit)

  # Check that values are rounded (no more than 3 decimal places)
  decimals <- sapply(as.vector(pcor), function(x) {
    if (x == 0) return(0)
    nchar(sub("^-?[0-9]+\\.?", "", as.character(x)))
  })
  expect_true(all(decimals <= 3))
})

