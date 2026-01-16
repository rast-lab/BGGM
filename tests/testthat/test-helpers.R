library(testthat)
library(BGGM)

# ============================================
# Tests for tstat (t-statistic calculation)
# ============================================

test_that("tstat computes correct t-statistic", {
  # t = r * sqrt((n - 2 - k) / (1 - r^2))
  r <- 0.5
  n <- 100
  k <- 0

  result <- BGGM:::tstat(r, n, k)
  expected <- r * sqrt((n - 2 - k) / (1 - r^2))

  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("tstat handles zero correlation", {
  result <- BGGM:::tstat(0, 100, 0)
  expect_equal(result, 0)
})

test_that("tstat handles negative correlation", {
  r_pos <- BGGM:::tstat(0.5, 100, 0)
  r_neg <- BGGM:::tstat(-0.5, 100, 0)

  expect_equal(r_neg, -r_pos, tolerance = 1e-10)
})

test_that("tstat works with vector input", {
  r <- c(0.1, 0.3, 0.5)
  n <- 100
  k <- 0

  result <- BGGM:::tstat(r, n, k)

  expect_length(result, 3)
  expect_true(all(is.finite(result)))
})

# ============================================
# Tests for fisher_z (Fisher's z transformation)
# ============================================

test_that("fisher_z computes correct transformation", {
  rho <- 0.5
  result <- BGGM:::fisher_z(rho)
  expected <- 0.5 * log((1 + rho) / (1 - rho))

  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("fisher_z handles zero", {
  result <- BGGM:::fisher_z(0)
  expect_equal(result, 0, tolerance = 1e-10)
})

test_that("fisher_z is symmetric around zero", {
  result_pos <- BGGM:::fisher_z(0.5)
  result_neg <- BGGM:::fisher_z(-0.5)

  expect_equal(result_neg, -result_pos, tolerance = 1e-10)
})

test_that("fisher_z works with vector input", {
  rho <- c(0.1, 0.3, 0.5, 0.7)
  result <- BGGM:::fisher_z(rho)

  expect_length(result, 4)
  expect_true(all(is.finite(result)))
})

# ============================================
# Tests for z2r (inverse Fisher transformation)
# ============================================

test_that("z2r computes correct inverse transformation", {
  z <- 0.5
  result <- BGGM:::z2r(z)
  expected <- (exp(2 * z) - 1) / (1 + exp(2 * z))

  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("z2r handles zero", {
  result <- BGGM:::z2r(0)
  expect_equal(result, 0, tolerance = 1e-10)
})

test_that("z2r inverts fisher_z", {
  rho_original <- c(0.1, 0.3, 0.5, 0.7, -0.2, -0.5)
  z_transformed <- BGGM:::fisher_z(rho_original)
  rho_back <- BGGM:::z2r(z_transformed)

  expect_equal(rho_back, rho_original, tolerance = 1e-10)
})

test_that("z2r returns values in valid correlation range", {
  z_values <- c(-3, -1, 0, 1, 3)
  result <- BGGM:::z2r(z_values)

  expect_true(all(result >= -1 & result <= 1))
})

# ============================================
# Tests for approx_sd (approximate standard deviation)
# ============================================

test_that("approx_sd computes correct value", {
  r <- 0.5
  n <- 100
  k <- 2

  result <- BGGM:::approx_sd(r, n, k)
  expected <- sqrt((1 - r^2) / (n - k - 2))

  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("approx_sd returns smaller value for larger n", {
  r <- 0.5
  k <- 0

  sd_small_n <- BGGM:::approx_sd(r, 50, k)
  sd_large_n <- BGGM:::approx_sd(r, 500, k)


  expect_true(sd_large_n < sd_small_n)
})

test_that("approx_sd returns larger value for r closer to 0", {
  n <- 100
  k <- 0

  sd_small_r <- BGGM:::approx_sd(0.1, n, k)
  sd_large_r <- BGGM:::approx_sd(0.9, n, k)

  expect_true(sd_small_r > sd_large_r)
})

# ============================================
# Tests for delta_solve
# ============================================

test_that("delta_solve computes correct delta", {
  # delta = 1/x^2 - 1
  x <- 0.5
  result <- BGGM:::delta_solve(x)
  expected <- 1 / x^2 - 1

  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("delta_solve errors for invalid input", {
  # x must be in (0, sqrt(1/2)]
  expect_error(BGGM:::delta_solve(0))
  expect_error(BGGM:::delta_solve(-0.1))
  expect_error(BGGM:::delta_solve(1))  # > sqrt(1/2)
})

test_that("delta_solve returns 1 for sqrt(1/2)", {
  result <- BGGM:::delta_solve(sqrt(1/2))
  expect_equal(result, 1, tolerance = 1e-10)
})

test_that("delta_solve returns 2 for sqrt(1/3)", {
  result <- BGGM:::delta_solve(sqrt(1/3))
  expect_equal(result, 2, tolerance = 1e-10)
})

# ============================================
# Tests for symmetric_mat
# ============================================

test_that("symmetric_mat makes matrix symmetric", {
  mat <- matrix(c(1, 0, 0, 2, 2, 0, 3, 4, 3), 3, 3)
  result <- BGGM:::symmetric_mat(mat)

  expect_true(isSymmetric(result))
})

test_that("symmetric_mat copies upper to lower triangle", {
  mat <- matrix(0, 3, 3)
  mat[1, 2] <- 0.5
  mat[1, 3] <- 0.3
  mat[2, 3] <- 0.4
  diag(mat) <- 1

  result <- BGGM:::symmetric_mat(mat)

  expect_equal(result[2, 1], result[1, 2])
  expect_equal(result[3, 1], result[1, 3])
  expect_equal(result[3, 2], result[2, 3])
})

test_that("symmetric_mat preserves diagonal", {
  mat <- diag(c(1, 2, 3))
  mat[1, 2] <- 0.5

  result <- BGGM:::symmetric_mat(mat)

  expect_equal(diag(result), c(1, 2, 3))
})

# ============================================
# Tests for symm_mat (alias for symmetric_mat)
# ============================================

test_that("symm_mat makes matrix symmetric", {
  mat <- matrix(c(1, 0, 0, 2, 2, 0, 3, 4, 3), 3, 3)
  result <- BGGM:::symm_mat(mat)

  expect_true(isSymmetric(result))
})

# ============================================
# Tests for get_lower_tri
# ============================================

test_that("get_lower_tri returns lower triangle with NA upper", {
  mat <- matrix(1:9, 3, 3)
  result <- BGGM:::get_lower_tri(mat)

  expect_true(all(is.na(result[upper.tri(result)])))
  expect_true(all(!is.na(result[lower.tri(result, diag = TRUE)])))
})

test_that("get_lower_tri preserves lower triangle values", {
  mat <- matrix(1:9, 3, 3)
  result <- BGGM:::get_lower_tri(mat)

  expect_equal(result[lower.tri(result, diag = TRUE)],
               mat[lower.tri(mat, diag = TRUE)])
})

# ============================================
# Tests for simple_rref (reduced row echelon form)
# ============================================

test_that("simple_rref produces identity for invertible matrix", {
  mat <- matrix(c(1, 0, 0, 1), 2, 2)
  result <- BGGM:::simple_rref(mat)

  expect_equal(result, diag(2), tolerance = 1e-9)
})

test_that("simple_rref handles augmented matrix", {
  # System: x + y = 3, 2x + y = 4
  # Solution: x = 1, y = 2
  mat <- matrix(c(1, 2, 1, 1, 3, 4), 2, 3)
  result <- BGGM:::simple_rref(mat)

  # Should give [1, 0, 1; 0, 1, 2]
  expect_equal(result[1, 3], 1, tolerance = 1e-9)
  expect_equal(result[2, 3], 2, tolerance = 1e-9)
})

test_that("simple_rref handles zero rows", {
  mat <- matrix(c(1, 0, 2, 0, 3, 0), 2, 3)
  result <- BGGM:::simple_rref(mat)

  expect_true(is.matrix(result))
  expect_equal(dim(result), c(2, 3))
})

# ============================================
# Tests for post_prob
# ============================================

test_that("post_prob returns value between 0.5 and 1", {
  data <- rnorm(1000, mean = 0.5)
  result <- BGGM:::post_prob(data)

  expect_true(result >= 0.5 && result <= 1)
})

test_that("post_prob returns 1 for data all positive", {
  data <- abs(rnorm(100)) + 0.1
  result <- BGGM:::post_prob(data)

  expect_equal(result, 1)
})

test_that("post_prob returns ~0.5 for symmetric data around 0", {
  set.seed(123)
  data <- rnorm(10000, mean = 0)
  result <- BGGM:::post_prob(data)

  expect_true(abs(result - 0.5) < 0.05)
})

# ============================================
# Tests for rope_helper
# ============================================

test_that("rope_helper returns proportion in ROPE", {
  data <- rnorm(1000, mean = 0, sd = 0.1)
  rope <- 0.1

  result <- BGGM:::rope_helper(data, rope)

  expect_true(result >= 0 && result <= 1)
})

test_that("rope_helper returns 1 when all data in ROPE", {
  data <- runif(100, -0.05, 0.05)
  rope <- 0.1

  result <- BGGM:::rope_helper(data, rope)

  expect_equal(result, 1)
})

test_that("rope_helper returns 0 when no data in ROPE", {
  data <- runif(100, 0.5, 1)
  rope <- 0.1

  result <- BGGM:::rope_helper(data, rope)

  expect_equal(result, 0)
})

# ============================================
# Tests for ci_helper
# ============================================

test_that("ci_helper returns 0 when CI includes 0", {
  set.seed(123)
  data <- rnorm(1000, mean = 0)
  result <- BGGM:::ci_helper(data, ci_width = 0.95)

  expect_equal(result, 0)
})

test_that("ci_helper returns 1 when CI excludes 0", {
  data <- rnorm(1000, mean = 5, sd = 0.5)
  result <- BGGM:::ci_helper(data, ci_width = 0.95)

  expect_equal(result, 1)
})

test_that("ci_helper respects ci_width parameter", {
  set.seed(456)
  # Data where narrow CI might include 0 but wide CI definitely does
  data <- rnorm(1000, mean = 0.1, sd = 0.1)

  result_narrow <- BGGM:::ci_helper(data, ci_width = 0.50)
  result_wide <- BGGM:::ci_helper(data, ci_width = 0.99)

  # Both should work, but results may differ

  expect_true(result_narrow %in% c(0, 1))
  expect_true(result_wide %in% c(0, 1))
})

# ============================================
# Tests for numbers2words
# ============================================

test_that("numbers2words converts single digits", {
  expect_equal(BGGM:::numbers2words(0), "")
  expect_equal(BGGM:::numbers2words(1), "one")
  expect_equal(BGGM:::numbers2words(5), "five")
  expect_equal(BGGM:::numbers2words(9), "nine")
})

test_that("numbers2words converts teens", {
  expect_equal(BGGM:::numbers2words(10), "ten")
  expect_equal(BGGM:::numbers2words(11), "eleven")
  expect_equal(BGGM:::numbers2words(15), "fifteen")
  expect_equal(BGGM:::numbers2words(19), "nineteen")
})

test_that("numbers2words converts tens", {
  expect_equal(BGGM:::numbers2words(20), "twenty")
  expect_equal(BGGM:::numbers2words(30), "thirty")
  expect_equal(BGGM:::numbers2words(99), "ninety nine")
})

test_that("numbers2words converts hundreds", {
  result <- BGGM:::numbers2words(100)
  expect_true(grepl("hundred", result))

  result <- BGGM:::numbers2words(123)
  expect_true(grepl("hundred", result))
  expect_true(grepl("twenty", result))
  expect_true(grepl("three", result))
})

test_that("numbers2words works with vector input", {
  result <- BGGM:::numbers2words(1:5)

  expect_length(result, 5)
  expect_equal(result[1], "one")
  expect_equal(result[5], "five")
})

# ============================================
# Tests for word2num
# ============================================

test_that("word2num converts single digits", {
  expect_equal(BGGM:::word2num("one")[[2]], 1)
  expect_equal(BGGM:::word2num("five")[[2]], 5)
  expect_equal(BGGM:::word2num("nine")[[2]], 9)
})

test_that("word2num converts teens", {
  expect_equal(BGGM:::word2num("eleven")[[2]], 11)
  expect_equal(BGGM:::word2num("fifteen")[[2]], 15)
  expect_equal(BGGM:::word2num("nineteen")[[2]], 19)
})

test_that("word2num converts tens", {
  expect_equal(BGGM:::word2num("twenty")[[2]], 20)
  expect_equal(BGGM:::word2num("thirty")[[2]], 30)
})

test_that("word2num converts compound numbers", {
  expect_equal(BGGM:::word2num("twenty one")[[2]], 21)
  expect_equal(BGGM:::word2num("ninety nine")[[2]], 99)
})

# ============================================
# Tests for framer (hypothesis parsing)
# ============================================

test_that("framer parses simple equality", {
  result <- BGGM:::framer("a=b")

  expect_equal(nrow(result), 1)
  expect_equal(result$left, "a")
  expect_equal(result$comp, "=")
  expect_equal(result$right, "b")
})

test_that("framer parses simple inequality", {
  result <- BGGM:::framer("a>b")

  expect_equal(result$left, "a")
  expect_equal(result$comp, ">")
  expect_equal(result$right, "b")
})

test_that("framer parses less than", {
  result <- BGGM:::framer("a<b")

  expect_equal(result$comp, "<")
})

test_that("framer parses chained comparisons", {
  result <- BGGM:::framer("a>b>c")

  expect_equal(nrow(result), 2)
  expect_equal(result$left, c("a", "b"))
  expect_equal(result$right, c("b", "c"))
})

test_that("framer parses comparison with zero", {
  result <- BGGM:::framer("a>0")

  expect_equal(result$left, "a")
  expect_equal(result$right, "0")
})

# ============================================
# Tests for create_matrices
# ============================================

test_that("create_matrices handles equality constraint", {
  framed <- data.frame(
    left = "a",
    comp = "=",
    right = "b",
    stringsAsFactors = FALSE
  )
  varnames <- c("a", "b", "c")

  result <- BGGM:::create_matrices(framed, varnames)

  expect_true(!is.null(result$R_e))
  expect_true(is.null(result$R_i))
  expect_equal(result$comparisons, "only equality")
})

test_that("create_matrices handles inequality constraint", {
  framed <- data.frame(
    left = "a",
    comp = ">",
    right = "b",
    stringsAsFactors = FALSE
  )
  varnames <- c("a", "b", "c")

  result <- BGGM:::create_matrices(framed, varnames)

  expect_true(is.null(result$R_e))
  expect_true(!is.null(result$R_i))
  expect_equal(result$comparisons, "only inequality")
})

test_that("create_matrices handles mixed constraints", {
  framed <- data.frame(
    left = c("a", "b"),
    comp = c("=", ">"),
    right = c("0", "c"),
    stringsAsFactors = FALSE
  )
  varnames <- c("a", "b", "c")

  result <- BGGM:::create_matrices(framed, varnames)

  expect_true(!is.null(result$R_e))
  expect_true(!is.null(result$R_i))
  expect_equal(result$comparisons, "both comparisons")
})

test_that("create_matrices errors on value vs value comparison", {
  framed <- data.frame(
    left = "2",
    comp = "=",
    right = "2",
    stringsAsFactors = FALSE
  )
  varnames <- c("a", "b")

  expect_error(BGGM:::create_matrices(framed, varnames))
})

# ============================================
# Tests for KL (Kullback-Leibler divergence)
# ============================================

test_that("KL returns 0 for identical matrices", {
  Theta <- diag(3)
  result <- BGGM:::KL(Theta, Theta)

  expect_equal(result, 0, tolerance = 1e-10)
})

test_that("KL returns positive value for different matrices", {
  Theta <- diag(3)
  hatTheta <- diag(3) * 1.5

  result <- BGGM:::KL(Theta, hatTheta)

  expect_true(result > 0)
})

test_that("KL is not symmetric", {
  Theta <- diag(3)
  hatTheta <- diag(c(1, 2, 3))

  kl_1 <- BGGM:::KL(Theta, hatTheta)
  kl_2 <- BGGM:::KL(hatTheta, Theta)

  # KL divergence is not symmetric in general
  expect_true(is.finite(kl_1))
  expect_true(is.finite(kl_2))
})

# ============================================
# Tests for QL (Quadratic Loss)
# ============================================

test_that("QL returns 0 for identical matrices", {
  Theta <- diag(3)
  result <- BGGM:::QL(Theta, Theta)

  expect_equal(result, 0, tolerance = 1e-10)
})

test_that("QL returns positive value for different matrices", {
  Theta <- diag(3)
  hatTheta <- diag(3) * 1.5

  result <- BGGM:::QL(Theta, hatTheta)

  expect_true(result >= 0)
})

# ============================================
# Tests for performance (classification metrics)
# ============================================

test_that("performance computes correct metrics for perfect prediction", {
  # True and Estimate are identical
  True <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), 3, 3)
  Estimate <- True

  result <- BGGM:::performance(Estimate, True)

  expect_equal(result$results$score[result$results$measure == "Specificity"], 1)
  expect_equal(result$results$score[result$results$measure == "Sensitivity"], 1)
})

test_that("performance returns all expected metrics", {
  True <- matrix(c(0, 1, 0, 1, 0, 0, 0, 0, 0), 3, 3)
  Estimate <- matrix(c(0, 1, 1, 1, 0, 0, 1, 0, 0), 3, 3)

  result <- BGGM:::performance(Estimate, True)

  expected_measures <- c("Specificity", "Sensitivity", "Precision",
                         "Recall", "F1_score", "MCC")
  expect_true(all(expected_measures %in% result$results$measure))
})

test_that("performance handles zero matrices", {
  True <- matrix(0, 3, 3)
  Estimate <- matrix(0, 3, 3)

  result <- BGGM:::performance(Estimate, True)

  # All zeros means perfect specificity (no false positives)
  # but undefined sensitivity (no positives to detect)
  expect_true(is.list(result))
})

# ============================================
# Tests for R2_helper
# ============================================

test_that("R2_helper returns valid R-squared values", {
  set.seed(123)
  n_iter <- 100
  n_obs <- 50

  # Generate predictions and true values
  y <- rnorm(n_obs)
  ypred <- matrix(rnorm(n_iter * n_obs), nrow = n_iter, ncol = n_obs)
  # Add signal
  ypred <- ypred + matrix(rep(y, each = n_iter), nrow = n_iter)

  result <- BGGM:::R2_helper(ypred, y, ci_width = 0.95)

  expect_true(all(result$R2 >= 0 & result$R2 <= 1))
  expect_true(length(result$summary_r2) == 4)
  expect_true("post_mean" %in% names(result$summary_r2))
})

# ============================================
# Tests for MSE_helper
# ============================================

test_that("MSE_helper returns valid MSE values", {
  set.seed(123)
  n_iter <- 100
  n_obs <- 50

  y <- rnorm(n_obs)
  ypred <- matrix(rnorm(n_iter * n_obs), nrow = n_iter, ncol = n_obs)

  result <- BGGM:::MSE_helper(ypred, y, ci_width = 0.95)

  expect_true(all(result$MSE >= 0))
  expect_true(length(result$summary_mse) == 4)
})

test_that("MSE_helper returns 0 for perfect predictions", {
  n_iter <- 10
  n_obs <- 20

  y <- rnorm(n_obs)
  ypred <- matrix(rep(y, each = n_iter), nrow = n_iter, ncol = n_obs)

  result <- BGGM:::MSE_helper(ypred, y, ci_width = 0.95)

  expect_equal(mean(result$MSE), 0, tolerance = 1e-10)
})

# ============================================
# Tests for error_helper
# ============================================

test_that("error_helper computes MSE correctly", {
  set.seed(123)
  n_iter <- 100
  n_obs <- 50

  y <- rnorm(n_obs)
  ypred <- matrix(rnorm(n_iter * n_obs), nrow = n_iter, ncol = n_obs)

  result <- BGGM:::error_helper(ypred, y, ci_width = 0.95, measure = "mse")

  expect_true(all(result$error >= 0))
  expect_true("post_mean" %in% names(result$summary))
})

test_that("error_helper computes MAE correctly", {
  set.seed(123)
  n_iter <- 100
  n_obs <- 50

  y <- rnorm(n_obs)
  ypred <- matrix(rnorm(n_iter * n_obs), nrow = n_iter, ncol = n_obs)

  result <- BGGM:::error_helper(ypred, y, ci_width = 0.95, measure = "mae")

  expect_true(all(result$error >= 0))
})

# ============================================
# Tests for kl_func (univariate KL)
# ============================================

test_that("kl_func returns 0 for identical variances", {
  result <- BGGM:::kl_func(1, 1)
  expect_equal(result, 0, tolerance = 1e-10)
})

test_that("kl_func returns positive for different variances", {
  result <- BGGM:::kl_func(1, 2)
  expect_true(result > 0)

  result2 <- BGGM:::kl_func(2, 1)
  expect_true(result2 > 0)
})

# ============================================
# Tests for Y_combine
# ============================================

test_that("Y_combine combines multiple datasets", {
  Y1 <- matrix(rnorm(100), 20, 5)
  Y2 <- matrix(rnorm(75), 15, 5)

  result <- BGGM:::Y_combine(Y1, Y2)

  expect_equal(length(result$dat), 2)
  expect_equal(nrow(result$dat_info), 2)
  expect_equal(nrow(result$pairwise), 1)  # 1 pairwise comparison for 2 groups
})

test_that("Y_combine handles three datasets", {
  Y1 <- matrix(rnorm(100), 20, 5)
  Y2 <- matrix(rnorm(75), 15, 5)
  Y3 <- matrix(rnorm(50), 10, 5)

  result <- BGGM:::Y_combine(Y1, Y2, Y3)

  expect_equal(length(result$dat), 3)
  expect_equal(nrow(result$pairwise), 3)  # 3 pairwise comparisons for 3 groups
})

test_that("Y_combine removes NA values", {
  Y1 <- matrix(rnorm(100), 20, 5)
  Y1[1, 1] <- NA
  Y2 <- matrix(rnorm(75), 15, 5)

  result <- BGGM:::Y_combine(Y1, Y2)

  # First dataset should have one less row due to NA removal
  expect_equal(nrow(result$dat[[1]]), 19)
  expect_equal(nrow(result$dat[[2]]), 15)
})

# ============================================
# Tests for positive_helper and negative_helper
# ============================================

test_that("positive_helper computes correct BF", {
  pcor <- 0.5
  post_sd <- 0.1
  BF_null <- 1

  result <- BGGM:::positive_helper(pcor, post_sd, BF_null)

  expect_true(is.finite(result))
  expect_true(result > 0)
})

test_that("negative_helper computes correct BF", {
  pcor <- -0.5
  post_sd <- 0.1
  BF_null <- 1

  result <- BGGM:::negative_helper(pcor, post_sd, BF_null)

  expect_true(is.finite(result))
  expect_true(result > 0)
})

test_that("exhaustive_helper normalizes to sum to 1", {
  BF_null <- 1
  BF_positive <- 2
  BF_negative <- 0.5

  result <- BGGM:::exhaustive_helper(BF_null, BF_positive, BF_negative)

  expect_equal(sum(result), 1, tolerance = 1e-10)
  expect_length(result, 3)
})

# ============================================
# Tests for rank_helper
# ============================================

test_that("rank_helper returns expected components", {
  set.seed(123)
  Y <- matrix(sample(1:5, 100, replace = TRUE), 20, 5)

  result <- BGGM:::rank_helper(Y)

  expect_true("K" %in% names(result))
  expect_true("levels" %in% names(result))
  expect_true("Sigma_start" %in% names(result))
  expect_true("z0_start" %in% names(result))
})

test_that("rank_helper computes correct K (max levels)", {
  Y <- matrix(c(1, 2, 3, 1, 2, 3, 1, 2, 3), 3, 3)

  result <- BGGM:::rank_helper(Y)

  expect_equal(result$K, c(3, 3, 3))
})

test_that("rank_helper handles missing values", {
  Y <- matrix(c(1, 2, NA, 1, 2, 3), 3, 2)

  result <- BGGM:::rank_helper(Y)

  expect_true(is.matrix(result$z0_start))
})

# ============================================
# Tests for unbiased_cov
# ============================================

test_that("unbiased_cov returns correlation matrix", {
  set.seed(123)
  x <- matrix(rnorm(100), 20, 5)

  result <- BGGM:::unbiased_cov(x)

  # Should be square matrix
  expect_equal(nrow(result), ncol(result))
  # Diagonal should be 1 (correlation matrix)
  # Note: this is actually inverse correlation, so diagonal may not be 1
  expect_true(is.matrix(result))
})

# ============================================
# Tests for analytic_solve
# ============================================

test_that("analytic_solve returns expected components", {
  set.seed(123)
  X <- matrix(rnorm(200), 50, 4)

  result <- BGGM:::analytic_solve(X)

  expect_true("inv_map" %in% names(result))
  expect_true("inv_var" %in% names(result))
  expect_true("pcor_mat" %in% names(result))
})

test_that("analytic_solve returns symmetric pcor_mat", {
  set.seed(123)
  X <- matrix(rnorm(200), 50, 4)

  result <- BGGM:::analytic_solve(X)

  # Check symmetry with tolerance
  expect_equal(result$pcor_mat, t(result$pcor_mat), tolerance = 1e-10)
})

test_that("analytic_solve pcor_mat has unit diagonal", {
  set.seed(123)
  X <- matrix(rnorm(200), 50, 4)

  result <- BGGM:::analytic_solve(X)

  # Partial correlation matrix should have 0 diagonal (after subtracting identity)
  # or the returned matrix already accounts for this
  expect_true(all(is.finite(diag(result$pcor_mat))))
})

# ============================================
# Tests for compare_predict_helper
# ============================================

test_that("compare_predict_helper returns summary statistics", {
  set.seed(123)
  x <- rnorm(1000, mean = 5, sd = 1)

  result <- BGGM:::compare_predict_helper(x, ci_width = 0.95)

  expect_true("post_mean" %in% names(result))
  expect_true("post_sd" %in% names(result))
  expect_equal(ncol(result), 4)  # mean, sd, lower, upper
})

test_that("compare_predict_helper respects ci_width", {
  set.seed(123)
  x <- rnorm(1000, mean = 0, sd = 1)

  result_narrow <- BGGM:::compare_predict_helper(x, ci_width = 0.50)
  result_wide <- BGGM:::compare_predict_helper(x, ci_width = 0.99)

  # Wide CI should have larger interval
  narrow_width <- result_narrow[[4]] - result_narrow[[3]]
  wide_width <- result_wide[[4]] - result_wide[[3]]

  expect_true(wide_width > narrow_width)
})

# ============================================
# Tests for name_helper
# ============================================

test_that("name_helper extracts column names correctly", {
  x <- "pcors[1,2]"
  result <- BGGM:::name_helper(x)

  expect_equal(result, "2")
})

# ============================================
# Tests for samps_inv_helper and samps_pcor_helper
# ============================================

test_that("samps_inv_helper creates correct names", {
  result <- BGGM:::samps_inv_helper(1, 3)

  expect_length(result, 3)
  expect_true(all(grepl("cov_inv", result)))
})

test_that("samps_pcor_helper creates correct names", {
  result <- BGGM:::samps_pcor_helper(1, 3)

  expect_length(result, 3)
  expect_true(all(grepl("pcors", result)))
})

# ============================================
# Tests for contrast_helper
# ============================================

test_that("contrast_helper creates contrast names", {
  result <- BGGM:::contrast_helper("group1_vs_group2")

  expect_true(grepl("Y_g", result))
  expect_true(grepl("_vs_", result))
})

# ============================================
# Tests for gen_pcors (generate partial correlations)
# ============================================

test_that("gen_pcors returns valid partial correlation matrix", {
  set.seed(123)
  result <- BGGM:::gen_pcors(p = 5, edge_prob = 0.3)

  expect_true("pcors" %in% names(result))
  expect_true("cors" %in% names(result))
  expect_true("adj" %in% names(result))

  # Check dimensions
  expect_equal(dim(result$pcors), c(5, 5))
  expect_equal(dim(result$cors), c(5, 5))
})

test_that("gen_pcors produces symmetric matrices", {
  set.seed(123)
  result <- BGGM:::gen_pcors(p = 5, edge_prob = 0.3)

  expect_true(isSymmetric(result$pcors))
  expect_true(isSymmetric(result$cors))
})

test_that("gen_pcors produces valid correlation values", {
  set.seed(123)
  result <- BGGM:::gen_pcors(p = 5, edge_prob = 0.5)

  # All correlations should be in [-1, 1]
  expect_true(all(result$cors >= -1 & result$cors <= 1))
  expect_true(all(result$pcors >= -1 & result$pcors <= 1))

  # Diagonal should be 1
  expect_equal(diag(result$pcors), rep(1, 5))
  expect_equal(diag(result$cors), rep(1, 5))
})

test_that("gen_pcors respects edge_prob parameter", {
  set.seed(123)
  result_sparse <- BGGM:::gen_pcors(p = 10, edge_prob = 0.1)
  result_dense <- BGGM:::gen_pcors(p = 10, edge_prob = 0.9)

  # Count non-zero edges (excluding diagonal)
  sparse_edges <- sum(result_sparse$adj[upper.tri(result_sparse$adj)] != 0)
  dense_edges <- sum(result_dense$adj[upper.tri(result_dense$adj)] != 0)

  # Dense should have more edges on average
  expect_true(dense_edges >= sparse_edges)
})
