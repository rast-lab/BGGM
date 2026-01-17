library(testthat)
library(BGGM)

# ============================================
# Tests for print.BGGM()
# ============================================

test_that("print.BGGM works with estimate object", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)

  output <- capture.output(print(fit))

  expect_true(length(output) > 0)
  expect_true(any(grepl("BGGM", output)))
})

test_that("print.BGGM works with explore object", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)

  output <- capture.output(print(fit))

  expect_true(length(output) > 0)
  expect_true(any(grepl("BGGM", output)))
})

test_that("print.BGGM works with select.estimate object", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  output <- capture.output(print(sel))

  expect_true(length(output) > 0)
})

test_that("print.BGGM works with select.explore object", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  output <- capture.output(print(sel))

  expect_true(length(output) > 0)
})

test_that("print.BGGM works with coef object", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  coefs <- coef(fit, progress = FALSE)

  output <- capture.output(print(coefs))

  expect_true(length(output) > 0)
  expect_true(any(grepl("Coefficients", output)))
})

test_that("print.BGGM works with summary.coef object", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  coefs <- coef(fit, progress = FALSE)
  summ <- summary(coefs)

  output <- capture.output(print(summ))

  expect_true(length(output) > 0)
})

test_that("print.BGGM works with ggm_compare_estimate object", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)

  output <- capture.output(print(fit))

  expect_true(length(output) > 0)
})

test_that("print.BGGM works with select.ggm_compare_estimate object", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_estimate(Y1, Y2, iter = 100, progress = FALSE)
  sel <- select(fit)

  output <- capture.output(print(sel))

  expect_true(length(output) > 0)
})

test_that("print.BGGM works with var_estimate object", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)

  output <- capture.output(print(fit))

  expect_true(length(output) > 0)
})

test_that("print.BGGM works with select.var_estimate object", {
  set.seed(123)
  Y <- subset(BGGM::ifit, id == 1)[1:50, -1]

  fit <- var_estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  output <- capture.output(print(sel))

  expect_true(length(output) > 0)
})

test_that("print.BGGM works with pcor_sum object", {
  set.seed(123)
  Y <- matrix(rnorm(200), 40, 5)
  colnames(Y) <- paste0("V", 1:5)

  fit <- estimate(Y, iter = 100, progress = FALSE)
  sums <- pcor_sum(fit, relations = "V1--V2 + V1--V3")

  output <- capture.output(print(sums))

  expect_true(length(output) > 0)
})

test_that("print.BGGM works with regression_summary object", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]
  Y$gender <- BGGM::bfi$gender[1:50]

  fit <- estimate(Y, formula = ~ gender,
                  type = "continuous",
                  iter = 100,
                  progress = FALSE)

  summ <- regression_summary(fit)

  output <- capture.output(print(summ))

  expect_true(length(output) > 0)
  expect_true(any(grepl("Residual Correlation", output)))
})

test_that("print.BGGM works with prior_var object", {
  set.seed(123)
  n <- 50
  p <- 3
  Y <- matrix(rnorm(n * p), n, p)

  suppressMessages({
    fit <- prior_belief_var(Y, est_ggm = FALSE, progress = FALSE)
  })

  output <- capture.output(print(fit))

  expect_true(length(output) > 0)
})

test_that("print.BGGM works with prior_ggm object", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  prior <- matrix(1, 5, 5)
  prior[1, 2] <- prior[2, 1] <- 5

  suppressMessages({
    fit <- prior_belief_ggm(Y, prior = prior, iter = 100, progress = FALSE)
  })

  output <- capture.output(print(fit))

  expect_true(length(output) > 0)
})

test_that("print.BGGM works with impute_data object", {
  set.seed(123)
  Y <- as.matrix(BGGM::bfi[1:50, 1:5])
  # Create NA values in the matrix
  Y[1, 1] <- NA
  Y[5, 2] <- NA
  Y[10, 3] <- NA

  fit <- impute_data(Y, iter = 100, progress = FALSE)

  output <- capture.output(print(fit))

  expect_true(length(output) > 0)
})

test_that("print.BGGM works with ggm_compare_explore object", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_explore(Y1, Y2, iter = 100, progress = FALSE)

  output <- capture.output(print(fit))

  expect_true(length(output) > 0)
})

test_that("print.BGGM works with select.ggm_compare_explore object", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_explore(Y1, Y2, iter = 100, progress = FALSE)
  sel <- select(fit)

  output <- capture.output(print(sel))

  expect_true(length(output) > 0)
})

test_that("print.BGGM works with ggm_compare_ppc object", {
  set.seed(123)
  Y <- BGGM::bfi[, 1:5]
  Y1 <- subset(Y, BGGM::bfi$gender == 1)[1:50, ]
  Y2 <- subset(Y, BGGM::bfi$gender == 2)[1:50, ]

  fit <- ggm_compare_ppc(Y1, Y2, iter = 100, progress = FALSE)

  output <- capture.output(print(fit))

  expect_true(length(output) > 0)
})

test_that("print.BGGM works with ggm_search object", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- ggm_search(Y, iter = 100, progress = FALSE)

  output <- capture.output(print(fit))

  expect_true(length(output) > 0)
})

test_that("print.BGGM works with precision object", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- estimate(Y, iter = 100, progress = FALSE)
  prec <- precision(fit)

  output <- capture.output(print(prec))

  expect_true(length(output) > 0)
})

test_that("print.BGGM works with bma_posterior object", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- ggm_search(Y, iter = 100, progress = FALSE)
  bma <- bma_posterior(fit)

  output <- capture.output(print(bma))

  expect_true(length(output) > 0)
})

test_that("print.BGGM works with constrained object", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit)

  const <- constrained_posterior(fit, adj = sel$Adj_10, iter = 100, progress = FALSE)

  output <- capture.output(print(const))

  expect_true(length(output) > 0)
})

