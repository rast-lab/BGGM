library(testthat)
library(BGGM)

# Test data setup
# Uses genuine correlational structure (not independent noise) so that
# select() actually finds edges -- select-based plot methods build a
# `network` object from the selected edges and error on an empty graph
# (see "plot.select.estimate works" below), so exercising them meaningfully
# requires data with some real signal.
set.seed(123)
n <- 100
p <- 4
Sigma <- diag(p)
Sigma[1, 2] <- Sigma[2, 1] <- 0.6
Sigma[3, 4] <- Sigma[4, 3] <- 0.5
test_data <- MASS::mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
colnames(test_data) <- paste0("V", 1:p)

# Create estimate object for testing
fit <- estimate(test_data, iter = 250, progress = FALSE)

# ============================================
# Tests for convergence
# ============================================

test_that("convergence works with estimate object", {
  result <- convergence(fit)

  expect_true(is.list(result))
  expect_true(length(result) > 0)
  expect_true(all(vapply(result, function(x) inherits(x, "ggplot"), logical(1))))
})

test_that("convergence works with param parameter", {
  result <- convergence(fit, param = "V1--V2")

  expect_true(is.list(result))
  expect_true(inherits(result[[1]], "ggplot"))
})

test_that("convergence works with explore object", {
  fit_explore <- explore(test_data, progress = FALSE)

  result <- convergence(fit_explore)

  expect_true(is.list(result))
  expect_true(length(result) > 0)
})

test_that("convergence works with type = 'trace'", {
  result <- convergence(fit, type = "trace")

  expect_true(is.list(result))
  expect_true(inherits(result[[1]], "ggplot"))
})

test_that("convergence works with type = 'acf'", {
  result <- convergence(fit, type = "acf")

  expect_true(is.list(result))
  expect_true(inherits(result[[1]], "ggplot"))
})

# ============================================
# Tests for plot_prior
# ============================================

test_that("plot_prior works with default parameters", {
  result <- plot_prior()

  expect_s3_class(result, "ggplot")
})

test_that("plot_prior works with prior_sd parameter", {
  result <- plot_prior(prior_sd = 0.25)

  expect_s3_class(result, "ggplot")
})

test_that("plot_prior works with different prior_sd values", {
  result1 <- plot_prior(prior_sd = 0.1)
  result2 <- plot_prior(prior_sd = 0.5)

  expect_s3_class(result1, "ggplot")
  expect_s3_class(result2, "ggplot")
})

# ============================================
# Tests for plot methods on various objects
#
# Note: there is no plot.estimate / plot.explore / plot.ggm_compare_estimate
# method (confirmed against NAMESPACE's registered S3 methods) -- only
# plot.summary.estimate / plot.summary.explore /
# plot.summary.ggm_compare_estimate exist, so summary() must be called
# first. plot() on the bare fitted object dispatches to plot.default and
# errors ("'x' is a list, but does not have components 'x' and 'y'").
# ============================================

test_that("plot(summary(fit)) works for an estimate object", {
  result <- plot(summary(fit))

  expect_s3_class(result, "ggplot")
})

test_that("plot(summary(fit)) works for an explore object", {
  fit_explore <- explore(test_data, progress = FALSE)

  result <- plot(summary(fit_explore))

  expect_s3_class(result, "ggplot")
})

test_that("plot.select.estimate works", {
  sel <- select(fit)

  result <- plot(sel)

  expect_true(is.list(result))
  expect_s3_class(result$plt, "ggplot")
})

test_that("plot.select.explore works", {
  fit_explore <- explore(test_data, progress = FALSE)
  sel <- select(fit_explore)

  result <- plot(sel)

  expect_true(is.list(result))
})

# ============================================
# Tests for summary plot methods
# ============================================

test_that("plot.summary_estimate works", {
  summ <- summary(fit)

  result <- plot(summ)

  expect_s3_class(result, "ggplot")
})

test_that("plot.summary_explore works", {
  fit_explore <- explore(test_data, progress = FALSE)
  summ <- summary(fit_explore)

  result <- plot(summ)

  expect_s3_class(result, "ggplot")
})

# ============================================
# Tests for network visualization
# ============================================

test_that("plot with type = 'network' works for select.estimate", {
  sel <- select(fit)

  result <- plot(sel, type = "network")

  expect_true(is.list(result))
  expect_s3_class(result$plt, "ggplot")
})

# ============================================
# Tests for ggm_compare plot methods
# ============================================

test_that("plot.summary.ggm_compare_estimate works", {
  set.seed(456)
  Sigma_cmp <- diag(4)
  Sigma_cmp[1, 2] <- Sigma_cmp[2, 1] <- 0.6
  Sigma_cmp[3, 4] <- Sigma_cmp[4, 3] <- 0.5
  group1 <- MASS::mvrnorm(60, mu = rep(0, 4), Sigma = Sigma_cmp)
  group2 <- MASS::mvrnorm(60, mu = rep(0, 4), Sigma = Sigma_cmp)
  colnames(group1) <- colnames(group2) <- paste0("V", 1:4)

  fit_compare <- ggm_compare_estimate(group1, group2, iter = 100, progress = FALSE)

  # no plot.ggm_compare_estimate method exists -- summary() first, matching
  # the registered plot.summary.ggm_compare_estimate method
  result <- plot(summary(fit_compare))

  expect_true(is.list(result))
})

# ============================================
# Tests for predictability plot
# ============================================

test_that("plot.predictability works", {
  pred <- predictability(fit, progress = FALSE)

  result <- plot(pred)

  expect_s3_class(result, "ggplot")
})

# ============================================
# Tests for roll_your_own plot
# ============================================

test_that("plot.roll_your_own works", {
  mean_abs_pcor <- function(x) mean(abs(x[upper.tri(x)]))
  ryo <- roll_your_own(fit, FUN = mean_abs_pcor)

  result <- plot(ryo)

  expect_s3_class(result, "ggplot")
})
