library(testthat)
library(BGGM)

# Test data setup
set.seed(123)
test_data_cont <- matrix(rnorm(100), ncol = 5)
colnames(test_data_cont) <- paste0("V", 1:5)

test_data_binary <- matrix(as.integer(rnorm(100) > 0), ncol = 5)
colnames(test_data_binary) <- paste0("V", 1:5)

test_data_ordinal <- matrix(sample(1:4, 100, replace = TRUE), ncol = 5)
colnames(test_data_ordinal) <- paste0("V", 1:5)

# Basic functionality tests
test_that("explore with default parameters works for continuous data", {
  result <- explore(test_data_cont, progress = FALSE)

  expect_s3_class(result, c("BGGM", "explore", "default"))
  expect_true(is.list(result))
  expect_true("pcor_mat" %in% names(result))
})

test_that("explore returns correct class for continuous type", {
  result <- explore(test_data_cont, type = "continuous", progress = FALSE)

  expect_s3_class(result, "explore")
  expect_identical(result$type, "continuous")
})

test_that("explore returns correct structure", {
  result <- explore(test_data_cont, progress = FALSE)

  expect_true(is.list(result))
  expect_true("pcor_mat" %in% names(result))
  expect_true("post_samp" %in% names(result))
  expect_true(is.matrix(result$pcor_mat))
  expect_equal(dim(result$pcor_mat), c(5, 5))
})

# Test different data types
test_that("explore handles binary data", {
  result <- explore(test_data_binary, type = "binary", iter = 10, progress = FALSE)

  expect_s3_class(result, "explore")
  expect_identical(result$type, "binary")
  expect_equal(dim(result$pcor_mat), c(5, 5))
})

test_that("explore handles ordinal data", {
  result <- explore(test_data_ordinal, type = "ordinal", iter = 10, progress = FALSE)

  expect_s3_class(result, "explore")
  expect_identical(result$type, "ordinal")
  expect_equal(dim(result$pcor_mat), c(5, 5))
})

test_that("explore handles mixed data type", {
  set.seed(456)
  mixed_data <- data.frame(
    cont = rnorm(30),
    bin = as.integer(rnorm(30) > 0),
    ord = sample(1:3, 30, replace = TRUE)
  )

  result <- explore(mixed_data, type = "mixed", iter = 10, progress = FALSE)

  expect_s3_class(result, "explore")
  expect_identical(result$type, "mixed")
})

# Test with formula (control variables)
test_that("explore handles formula for control variables", {
  set.seed(789)
  Y <- matrix(rnorm(100), ncol = 4)
  colnames(Y) <- paste0("node", 1:4)
  control_var <- rnorm(25)
  dat <- data.frame(Y, control = control_var)

  result <- explore(dat, formula = ~control, progress = FALSE)

  expect_s3_class(result, "explore")
  expect_equal(result$p, 4)
  expect_true("X" %in% names(result))
})

# Test error handling
test_that("explore rejects invalid type", {
  expect_error(explore(test_data_cont, type = "invalid", progress = FALSE))
})

# Test prior specification
test_that("explore accepts prior_sd parameter", {
  result <- explore(test_data_cont, prior_sd = 0.5, progress = FALSE)

  expect_s3_class(result, "explore")
})

# Note: analytic = TRUE is not available for explore (only for estimate)

# Test summary method
test_that("summary.explore returns expected output", {
  fit <- explore(test_data_cont, progress = FALSE)
  summ <- summary(fit)

  expect_s3_class(summ, "summary_explore")
  expect_true(is.data.frame(summ$dat_results))
})

# Test plot method
test_that("plot.summary_explore works without error", {
  fit <- explore(test_data_cont, progress = FALSE)
  summ <- summary(fit)

  expect_silent(plot(summ))
})

# Test posterior samples structure
test_that("explore posterior samples have correct structure", {
  result <- explore(test_data_cont, iter = 100, progress = FALSE)

  expect_true(is.array(result$post_samp$pcors))
  # Posterior samples should have positive number of samples
  expect_true(dim(result$post_samp$pcors)[3] > 0)
})

# Test with subset of data
test_that("explore handles small datasets", {
  small_data <- matrix(rnorm(40), ncol = 4)
  colnames(small_data) <- paste0("V", 1:4)

  result <- explore(small_data, progress = FALSE)

  expect_s3_class(result, "explore")
  expect_equal(dim(result$pcor_mat), c(4, 4))
})

test_that("matrix-F epsilon shrinks with the number of variables", {
  expect_equal(BGGM:::eps_default(5), 0.01)
  expect_equal(BGGM:::eps_default(10), 0.01)
  expect_equal(BGGM:::eps_default(200), 1 / 2000)
  # nu = 1 / eps must exceed p - 1
  for (p in c(2, 50, 150, 1000)) expect_gt(1 / BGGM:::eps_default(p), p - 1)

  set.seed(1)
  Y <- matrix(rnorm(40 * 20), 40, 20)
  fit <- explore(Y, iter = 50, progress = FALSE)
  expect_equal(fit$eps, 1 / 200)
})

test_that("explore runs when n < p", {
  set.seed(1)
  Y <- matrix(rnorm(15 * 20), 15, 20)
  fit <- explore(Y, iter = 50, progress = FALSE)
  expect_false(any(is.na(fit$pcor_mat)))
})

test_that("analytic prior sd of Fisher z matches the marginal beta prior", {
  expect_equal(BGGM:::prior_sd_z(1), pi / 2, tolerance = 1e-6)
  set.seed(1)
  r <- 2 * rbeta(2e5, 1.5, 1.5) - 1
  expect_equal(BGGM:::prior_sd_z(3), sd(atanh(r)), tolerance = 0.01)

  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 50, progress = FALSE)
  expect_null(fit$prior_samp)
  expect_equal(fit$prior_sd_z, BGGM:::prior_sd_z(3))
})

test_that("store_prior_draws = TRUE returns joint prior draws", {
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 50, progress = FALSE, store_prior_draws = TRUE)
  expect_equal(dim(fit$prior_samp$fisher_z), c(5, 5, 100))
  expect_equal(dim(fit$prior_samp$pcors), c(5, 5, 100))
  # selection still uses the analytic prior sd
  a <- select(fit)
  fit$prior_samp <- NULL
  expect_equal(a$BF_10, select(fit)$BF_10)
})

test_that("running summaries match the stored draws", {
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 200, progress = FALSE, seed = 1)
  idx <- 51:250
  expect_equal(fit$post_samp$z_mean,
               apply(fit$post_samp$fisher_z[,, idx], 1:2, mean), tolerance = 1e-8)
  expect_equal(fit$post_samp$z_sd,
               apply(fit$post_samp$fisher_z[,, idx], 1:2, sd), tolerance = 1e-8)
  expect_equal(fit$post_samp$pcor_sd,
               apply(fit$post_samp$pcors[,, idx], 1:2, sd), tolerance = 1e-8)
  expect_equal(fit$pcor_mat,
               apply(fit$post_samp$pcors[,, idx], 1:2, mean), tolerance = 1e-8)
})

test_that("store_post_draws = FALSE drops the draws; select() and summary() still work", {
  Y <- BGGM::bfi[1:100, 1:5]
  fit_t <- explore(Y, iter = 200, progress = FALSE, seed = 1)
  fit_f <- explore(Y, iter = 200, progress = FALSE, seed = 1, store_post_draws = FALSE)

  expect_null(fit_f$post_samp$pcors)
  expect_null(fit_f$post_samp$fisher_z)
  expect_false(fit_f$store_post_draws)
  expect_equal(fit_t$pcor_mat, fit_f$pcor_mat)

  for (alt in c("two.sided", "greater", "less", "exhaustive")) {
    a <- select(fit_t, alternative = alt)
    b <- select(fit_f, alternative = alt)
    expect_equal(a$pcor_mat, b$pcor_mat, tolerance = 1e-6)
    expect_equal(a$incl_prob, b$incl_prob, tolerance = 1e-6)
    a <- select(fit_t, method = "BMA", alternative = alt)
    b <- select(fit_f, method = "BMA", alternative = alt)
    expect_equal(a$pcor_mat_zero, b$pcor_mat_zero, tolerance = 1e-6)
  }
  expect_equal(summary(fit_t)$dat_results, summary(fit_f)$dat_results)
  expect_equal(summary(select(fit_t))$summary, summary(select(fit_f))$summary)
})

test_that("store_post_draws = FALSE works for other data types", {
  fit <- explore(test_data_binary, type = "binary", iter = 100,
                 progress = FALSE, store_post_draws = FALSE)
  expect_null(fit$post_samp$pcors)
  expect_s3_class(select(fit), "select.explore")

  fit <- explore(test_data_ordinal, type = "ordinal", iter = 100,
                 progress = FALSE, store_post_draws = FALSE)
  expect_null(fit$post_samp$pcors)
  expect_s3_class(select(fit), "select.explore")

  fit <- explore(test_data_ordinal, type = "mixed", iter = 100,
                 progress = FALSE, store_post_draws = FALSE)
  expect_null(fit$post_samp$pcors)
  expect_s3_class(select(fit), "select.explore")

  dat <- data.frame(test_data_cont[, 1:4], control = rnorm(20))
  fit <- explore(dat, formula = ~ control, iter = 100,
                 progress = FALSE, store_post_draws = FALSE)
  expect_null(fit$post_samp$pcors)
  expect_false(is.null(fit$post_samp$beta))
  expect_s3_class(select(fit), "select.explore")
})

test_that("functions that need draws give a clear error", {
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE, store_post_draws = FALSE)
  expect_error(posterior_samples(fit), "store_post_draws = TRUE")
  expect_error(pcor_to_cor(fit), "store_post_draws = TRUE")
  expect_error(coef(fit), "store_post_draws = TRUE")
  expect_error(convergence(fit), "store_post_draws = TRUE")
  expect_error(predictability(fit), "store_post_draws = TRUE")
  expect_error(predict(fit), "store_post_draws = TRUE")
})
