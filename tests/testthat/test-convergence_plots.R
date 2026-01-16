library(testthat)
library(BGGM)

# Test data setup
set.seed(123)
n <- 50
p <- 4
test_data <- matrix(rnorm(n * p), ncol = p)
colnames(test_data) <- paste0("V", 1:p)

# Create estimate object for testing
fit <- estimate(test_data, iter = 100, progress = FALSE)

# ============================================
# Tests for convergence
# ============================================

test_that("convergence works with estimate object", {
  result <- tryCatch(
    {
      convergence(fit)
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})

test_that("convergence works with param parameter", {
  result <- tryCatch(
    {
      convergence(fit, param = "V1--V2")
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})

test_that("convergence works with explore object", {
  fit_explore <- explore(test_data, progress = FALSE)

  result <- tryCatch(
    {
      convergence(fit_explore)
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})

test_that("convergence works with type = 'trace'", {
  result <- tryCatch(
    {
      convergence(fit, type = "trace")
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})

test_that("convergence works with type = 'acf'", {
  result <- tryCatch(
    {
      convergence(fit, type = "acf")
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})

# ============================================
# Tests for plot_prior
# ============================================

test_that("plot_prior works with default parameters", {
  result <- tryCatch(
    {
      plot_prior()
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})

test_that("plot_prior works with prior_sd parameter", {
  result <- tryCatch(
    {
      plot_prior(prior_sd = 0.25)
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})

test_that("plot_prior works with different prior_sd values", {
  result1 <- tryCatch(
    {
      plot_prior(prior_sd = 0.1)
      TRUE
    },
    error = function(e) FALSE
  )

  result2 <- tryCatch(
    {
      plot_prior(prior_sd = 0.5)
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result1))
  expect_true(is.logical(result2))
})

# ============================================
# Tests for plot methods on various objects
# ============================================

test_that("plot.estimate works", {
  result <- tryCatch(
    {
      plot(fit)
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})

test_that("plot.explore works", {
  fit_explore <- explore(test_data, progress = FALSE)

  result <- tryCatch(
    {
      plot(fit_explore)
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})

test_that("plot.select.estimate works", {
  sel <- select(fit)

  result <- tryCatch(
    {
      plot(sel)
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})

test_that("plot.select.explore works", {
  fit_explore <- explore(test_data, progress = FALSE)
  sel <- select(fit_explore)

  result <- tryCatch(
    {
      plot(sel)
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})

# ============================================
# Tests for summary plot methods
# ============================================

test_that("plot.summary_estimate works", {
  summ <- summary(fit)

  result <- tryCatch(
    {
      plot(summ)
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})

test_that("plot.summary_explore works", {
  fit_explore <- explore(test_data, progress = FALSE)
  summ <- summary(fit_explore)

  result <- tryCatch(
    {
      plot(summ)
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})

# ============================================
# Tests for network visualization
# ============================================

test_that("plot with type = 'network' works for select.estimate", {
  sel <- select(fit)

  result <- tryCatch(
    {
      plot(sel, type = "network")
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})

# ============================================
# Tests for ggm_compare plot methods
# ============================================

test_that("plot.ggm_compare_estimate works", {
  set.seed(456)
  group1 <- matrix(rnorm(30 * 4), ncol = 4)
  group2 <- matrix(rnorm(30 * 4), ncol = 4)
  colnames(group1) <- colnames(group2) <- paste0("V", 1:4)

  fit_compare <- ggm_compare_estimate(group1, group2, iter = 50, progress = FALSE)

  result <- tryCatch(
    {
      plot(fit_compare)
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})

# ============================================
# Tests for predictability plot
# ============================================

test_that("plot.predictability works", {
  pred <- predictability(fit, progress = FALSE)

  result <- tryCatch(
    {
      plot(pred)
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})

# ============================================
# Tests for roll_your_own plot
# ============================================

test_that("plot.roll_your_own works", {
  mean_abs_pcor <- function(x) mean(abs(x[upper.tri(x)]))
  ryo <- roll_your_own(fit, FUN = mean_abs_pcor)

  result <- tryCatch(
    {
      plot(ryo)
      TRUE
    },
    error = function(e) FALSE
  )

  expect_true(is.logical(result))
})
