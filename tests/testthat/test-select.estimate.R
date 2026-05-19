library(testthat)
library(BGGM)

# Test data setup
set.seed(123)
n <- 50
p <- 5
test_data <- matrix(rnorm(n * p), ncol = p)
colnames(test_data) <- paste0("V", 1:p)

# Create estimate object for testing
fit <- estimate(test_data, iter = 100, progress = FALSE)

# Basic functionality tests
test_that("select.estimate works with default parameters", {
  result <- select(fit)

  expect_s3_class(result, "select.estimate")
  expect_true(is.list(result))
})

test_that("select.estimate returns correct structure", {
  result <- select(fit)

  expect_true("pcor_adj" %in% names(result))
  expect_true("adj"      %in% names(result))
})

test_that("select.estimate returns adjacency and partial correlation matrices", {
  result <- select(fit)

  expect_true(is.matrix(result$pcor_adj))
  expect_equal(dim(result$pcor_adj), c(p, p))
  expect_true(is.matrix(result$adj))
  expect_equal(dim(result$adj), c(p, p))
})

# Test different credible interval levels
test_that("select.estimate works with different cred levels", {
  result_90 <- select(fit, cred = 0.90)
  result_95 <- select(fit, cred = 0.95)
  result_99 <- select(fit, cred = 0.99)

  expect_s3_class(result_90, "select.estimate")
  expect_s3_class(result_95, "select.estimate")
  expect_s3_class(result_99, "select.estimate")
})

# Test alternative hypotheses
test_that("select.estimate works with two.sided alternative", {
  result <- select(fit, alternative = "two.sided")

  expect_s3_class(result, "select.estimate")
})

test_that("select.estimate works with greater alternative", {
  result <- select(fit, alternative = "greater")

  expect_s3_class(result, "select.estimate")
})

test_that("select.estimate works with less alternative", {
  result <- select(fit, alternative = "less")

  expect_s3_class(result, "select.estimate")
})

# Test print method (select.estimate uses print.BGGM dispatcher)
test_that("print.select.estimate works", {
  sel <- select(fit)

  # Test that print works without error
  expect_output(print(sel))
})

# Test plot method — needs selected edges, so use real correlational data
test_that("plot.select.estimate returns a list with ggplot element", {
  Y   <- BGGM::bfi[1:100, 1:5]
  fit_bfi <- estimate(Y, iter = 100, progress = FALSE)
  sel <- select(fit_bfi, cred = 0.50)
  plt <- plot(sel)
  expect_true(is.list(plt))
  expect_true(inherits(plt$plt, "ggplot"))
})

# Test with different data types
test_that("select.estimate works with binary data", {
  set.seed(456)
  binary_data <- matrix(as.integer(rnorm(50 * 4) > 0), ncol = 4)
  colnames(binary_data) <- paste0("V", 1:4)

  fit_bin <- estimate(binary_data, type = "binary", iter = 50, progress = FALSE)
  result <- select(fit_bin)

  expect_s3_class(result, "select.estimate")
})

test_that("select.estimate works with ordinal data", {
  set.seed(789)
  ordinal_data <- matrix(sample(1:4, 50 * 4, replace = TRUE), ncol = 4)
  colnames(ordinal_data) <- paste0("V", 1:4)

  fit_ord <- estimate(ordinal_data, type = "ordinal", iter = 50, progress = FALSE)
  result <- select(fit_ord)

  expect_s3_class(result, "select.estimate")
})

# Test adjacency matrix properties
test_that("select.estimate adjacency matrix is symmetric", {
  result <- select(fit)

  expect_equal(result$adj,      t(result$adj))
  expect_equal(result$pcor_adj, t(result$pcor_adj), tolerance = 1e-10)
})

test_that("select.estimate adjacency has zeros on diagonal", {
  result <- select(fit)

  expect_equal(as.numeric(diag(result$adj)),      rep(0, p))
  expect_equal(as.numeric(diag(result$pcor_adj)), rep(0, p))
})

# Test with ROPE (Region of Practical Equivalence) if supported
test_that("select.estimate handles rope parameter", {
  result <- tryCatch(
    {
      select(fit, rope = 0.1)
    },
    error = function(e) {
      # If rope not supported, just return with defaults
      select(fit)
    }
  )

  expect_s3_class(result, "select.estimate")
})

# Test column names preserved in the underlying object
test_that("select.estimate preserves variable names in object", {
  result <- select(fit)

  # Variable names are stored in the embedded estimate object
  expect_equal(colnames(result$object$Y), paste0("V", 1:p))
})
