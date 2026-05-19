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
# Tests for pcor_to_cor
# ============================================

test_that("pcor_to_cor works with estimate object", {
  result <- pcor_to_cor(fit)

  expect_true(is.list(result))
})

test_that("pcor_to_cor returns R and R_mean", {
  result <- pcor_to_cor(fit)

  expect_true("R" %in% names(result))
  expect_true("R_mean" %in% names(result))
})

test_that("pcor_to_cor R is a 3D array", {
  result <- pcor_to_cor(fit)

  expect_true(is.array(result$R))
  expect_equal(length(dim(result$R)), 3)
  expect_equal(dim(result$R)[1], p)
  expect_equal(dim(result$R)[2], p)
})

test_that("pcor_to_cor R_mean is symmetric matrix", {
  result <- pcor_to_cor(fit)

  R_mean <- result$R_mean
  expect_true(is.matrix(R_mean))
  expect_equal(R_mean, t(R_mean), tolerance = 1e-10)
})

test_that("pcor_to_cor R_mean has ones on diagonal", {
  result <- pcor_to_cor(fit)

  R_mean <- result$R_mean
  expect_equal(as.numeric(diag(R_mean)), rep(1, p), tolerance = 1e-10)
})

test_that("pcor_to_cor R_mean values are in [-1, 1]", {
  result <- pcor_to_cor(fit)

  R_mean <- result$R_mean
  expect_true(all(R_mean >= -1 & R_mean <= 1))
})

test_that("pcor_to_cor works with explore object", {
  fit_explore <- explore(test_data, progress = FALSE)
  result <- pcor_to_cor(fit_explore)

  expect_true(is.list(result))
  expect_true("R" %in% names(result))
  expect_true("R_mean" %in% names(result))
})

# ============================================
# Tests for zero_order_cors
# ============================================

test_that("zero_order_cors works with continuous data", {
  result <- zero_order_cors(test_data, type = "continuous",
                            iter = 50, progress = FALSE)

  expect_true(is.list(result))
  expect_true("R" %in% names(result))
})

test_that("zero_order_cors R_mean is symmetric", {
  result <- zero_order_cors(test_data, type = "continuous",
                            iter = 50, progress = FALSE)

  expect_equal(result$R_mean, t(result$R_mean), tolerance = 1e-10)
})

test_that("zero_order_cors works with binary data", {
  set.seed(456)
  binary_data <- matrix(as.integer(rnorm(50 * 4) > 0), ncol = 4)
  colnames(binary_data) <- paste0("V", 1:4)

  # For binary data, use type = "binary"
  result <- zero_order_cors(binary_data, type = "binary",
                            iter = 50, progress = FALSE)

  expect_true(is.list(result))
})

test_that("zero_order_cors works with ordinal data", {
  set.seed(789)
  ordinal_data <- matrix(sample(1:5, 50 * 4, replace = TRUE), ncol = 4)
  colnames(ordinal_data) <- paste0("V", 1:4)

  # For ordinal data, use type = "ordinal"
  result <- zero_order_cors(ordinal_data, type = "ordinal",
                            iter = 50, progress = FALSE)

  expect_true(is.list(result))
})

test_that("zero_order_cors works with mixed type", {
  result <- zero_order_cors(test_data, type = "mixed",
                            iter = 50, progress = FALSE)

  expect_true(is.list(result))
})

# ============================================
# Tests for weighted_adj_mat
# ============================================
# Note: weighted_adj_mat requires a select object, not raw estimate/explore

test_that("weighted_adj_mat works with select.estimate object", {
  sel <- select(fit)
  result <- weighted_adj_mat(sel)

  expect_true(is.matrix(result))
})

test_that("weighted_adj_mat returns correct dimensions", {
  sel <- select(fit)
  result <- weighted_adj_mat(sel)

  expect_equal(dim(result), c(p, p))
})

test_that("weighted_adj_mat is symmetric", {
  sel <- select(fit)
  result <- weighted_adj_mat(sel)

  expect_equal(result, t(result), tolerance = 1e-10)
})

test_that("weighted_adj_mat has zeros on diagonal", {
  sel <- select(fit)
  result <- weighted_adj_mat(sel)

  expect_equal(as.numeric(diag(result)), rep(0, p))
})

test_that("weighted_adj_mat errors on select.explore object", {
  fit_explore <- explore(test_data, progress = FALSE)
  sel <- select(fit_explore)

  expect_error(weighted_adj_mat(sel), "weighted adjacency matrix not found")
})

# ============================================
# Tests for precision
# ============================================
# Note: precision() returns a list with precision_mean and precision (3D array)

test_that("precision works with estimate object", {
  result <- precision(fit, progress = FALSE)

  expect_true(is.list(result))
  expect_s3_class(result, "precision")
})

test_that("precision returns precision_mean and precision", {
  result <- precision(fit, progress = FALSE)

  expect_true("precision_mean" %in% names(result))
  expect_true("precision" %in% names(result))
})

test_that("precision_mean has correct dimensions", {
  result <- precision(fit, progress = FALSE)

  expect_true(is.matrix(result$precision_mean))
  expect_equal(dim(result$precision_mean), c(p, p))
})

test_that("precision_mean is symmetric", {
  result <- precision(fit, progress = FALSE)

  expect_equal(result$precision_mean, t(result$precision_mean), tolerance = 1e-10)
})

test_that("precision_mean has positive diagonal", {
  result <- precision(fit, progress = FALSE)

  # Precision matrix diagonal elements should be positive
  expect_true(all(diag(result$precision_mean) > 0))
})

test_that("precision array has correct dimensions", {
  result <- precision(fit, progress = FALSE)

  # precision should be a 3D array (p x p x iter)
  expect_true(is.array(result$precision))
  expect_equal(dim(result$precision)[1], p)
  expect_equal(dim(result$precision)[2], p)
})

# ============================================
# Tests for pcor_mat
# ============================================

test_that("pcor_mat works with estimate object", {
  result <- pcor_mat(fit)

  expect_true(is.matrix(result))
})

test_that("pcor_mat returns correct dimensions", {
  result <- pcor_mat(fit)

  expect_equal(dim(result), c(p, p))
})

test_that("pcor_mat is symmetric", {
  result <- pcor_mat(fit)

  expect_equal(result, t(result), tolerance = 1e-10)
})

test_that("pcor_mat has zeros on diagonal", {
  result <- pcor_mat(fit)

  # Use as.numeric to strip names
  expect_equal(as.numeric(diag(result)), rep(0, p))
})

test_that("pcor_mat values are in valid range", {
  result <- pcor_mat(fit)

  # Partial correlations should be in [-1, 1]
  expect_true(all(result >= -1 & result <= 1))
})

test_that("pcor_mat works with explore object", {
  fit_explore <- explore(test_data, progress = FALSE)
  result <- pcor_mat(fit_explore)

  expect_true(is.matrix(result))
})

test_that("pcor_mat preserves column names", {
  result <- pcor_mat(fit)

  expect_equal(colnames(result), paste0("V", 1:p))
  expect_equal(rownames(result), paste0("V", 1:p))
})
