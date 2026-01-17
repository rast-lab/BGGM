library(testthat)
library(BGGM)

# ============================================
# Tests for regression_summary()
# ============================================

test_that("regression_summary returns correct class", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]
  Y$gender <- BGGM::bfi$gender[1:50]

  fit <- estimate(Y, formula = ~ gender,
                  type = "continuous",
                  iter = 100,
                  progress = FALSE)

  summ <- regression_summary(fit)

  expect_s3_class(summ, "BGGM")
  expect_s3_class(summ, "regression_summary")
})

test_that("regression_summary returns expected components", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]
  Y$gender <- BGGM::bfi$gender[1:50]

  fit <- estimate(Y, formula = ~ gender,
                  type = "continuous",
                  iter = 100,
                  progress = FALSE)

  summ <- regression_summary(fit)

  expect_true("reg_summary" %in% names(summ))
  expect_true("resid_cor" %in% names(summ))
  expect_true("object" %in% names(summ))
})

test_that("regression_summary reg_summary has correct length", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]
  Y$gender <- BGGM::bfi$gender[1:50]

  fit <- estimate(Y, formula = ~ gender,
                  type = "continuous",
                  iter = 100,
                  progress = FALSE)

  summ <- regression_summary(fit)

  # Should have 5 regression summaries (one per outcome)
  expect_equal(length(summ$reg_summary), 5)
})

test_that("regression_summary reg_summary contains correct columns", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]
  Y$gender <- BGGM::bfi$gender[1:50]

  fit <- estimate(Y, formula = ~ gender,
                  type = "continuous",
                  iter = 100,
                  progress = FALSE)

  summ <- regression_summary(fit)

  # Check column names
  expected_cols <- c("Post.mean", "Post.sd", "Cred.lb", "Cred.ub")
  expect_equal(colnames(summ$reg_summary[[1]]), expected_cols)
})

test_that("regression_summary resid_cor has correct dimensions", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]
  Y$gender <- BGGM::bfi$gender[1:50]

  fit <- estimate(Y, formula = ~ gender,
                  type = "continuous",
                  iter = 100,
                  progress = FALSE)

  summ <- regression_summary(fit)

  # Should be 5x5 correlation matrix
  expect_equal(dim(summ$resid_cor), c(5, 5))
})

test_that("regression_summary resid_cor is symmetric", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]
  Y$gender <- BGGM::bfi$gender[1:50]

  fit <- estimate(Y, formula = ~ gender,
                  type = "continuous",
                  iter = 100,
                  progress = FALSE)

  summ <- regression_summary(fit)

  expect_true(isSymmetric(summ$resid_cor))
})

test_that("regression_summary resid_cor diagonal is 1", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]
  Y$gender <- BGGM::bfi$gender[1:50]

  fit <- estimate(Y, formula = ~ gender,
                  type = "continuous",
                  iter = 100,
                  progress = FALSE)

  summ <- regression_summary(fit)

  # Diagonal should be close to 1
  expect_true(all(abs(diag(summ$resid_cor) - 1) < 0.01))
})

test_that("regression_summary with custom cred", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]
  Y$gender <- BGGM::bfi$gender[1:50]

  fit <- estimate(Y, formula = ~ gender,
                  type = "continuous",
                  iter = 100,
                  progress = FALSE)

  summ <- regression_summary(fit, cred = 0.90)

  expect_s3_class(summ, "regression_summary")
})

test_that("regression_summary errors for non-estimate object", {
  set.seed(123)
  Y <- matrix(rnorm(100), 20, 5)

  expect_error(
    regression_summary(Y),
    "must be an estimate object"
  )
})

test_that("regression_summary print method works", {
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
  expect_true(any(grepl("BGGM", output)))
  expect_true(any(grepl("Coefficients", output)))
  expect_true(any(grepl("Residual Correlation", output)))
})

test_that("regression_summary with multiple predictors", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  Y$gender <- BGGM::bfi$gender[1:100]

  fit <- estimate(Y, formula = ~ gender,
                  type = "continuous",
                  iter = 100,
                  progress = FALSE)

  summ <- regression_summary(fit)

  expect_s3_class(summ, "regression_summary")
  # Check that rownames include predictors
  expect_true(nrow(summ$reg_summary[[1]]) >= 2)
})

test_that("regression_summary preserves column names", {
  set.seed(123)
  Y <- BGGM::bfi[1:50, 1:5]
  Y$gender <- BGGM::bfi$gender[1:50]

  fit <- estimate(Y, formula = ~ gender,
                  type = "continuous",
                  iter = 100,
                  progress = FALSE)

  summ <- regression_summary(fit)

  # Names should match original column names
  expect_equal(names(summ$reg_summary), colnames(Y)[1:5])
})

test_that("regression_summary without column names", {
  set.seed(123)
  Y <- as.matrix(BGGM::bfi[1:50, 1:5])
  colnames(Y) <- NULL
  Y <- cbind(Y, gender = BGGM::bfi$gender[1:50])
  Y <- as.data.frame(Y)
  colnames(Y) <- c(paste0("V", 1:5), "gender")

  fit <- estimate(Y, formula = ~ gender,
                  type = "continuous",
                  iter = 100,
                  progress = FALSE)

  summ <- regression_summary(fit)

  expect_s3_class(summ, "regression_summary")
})

