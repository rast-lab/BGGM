library(testthat)
library(BGGM)

test_that("select.explore returns expected structure for two-sided alternative", {
  Y <- matrix(rnorm(100), ncol = 4)
  fit <- explore(Y, progress = FALSE)
  result <- select.explore(fit, alternative = "two.sided")
  
  expect_s3_class(result, "select.explore")
  expect_named(result, c("pcor_mat_zero", "pcor_mat", "pcor_sd_fisher", "Adj_10", "Adj_01", "BF_10", "BF_01", "BF_cut", "method", "alternative", "call", "type", "formula", "analytic", "object"))
  expect_true(is.matrix(result$pcor_mat_zero))
  expect_true(is.matrix(result$pcor_mat))
  expect_true(is.matrix(result$Adj_10))
  expect_true(is.matrix(result$Adj_01))
})

test_that("select.explore returns expected structure for greater alternative", {
  Y <- matrix(rnorm(100), ncol = 4)
  fit <- explore(Y, progress = FALSE)
  result <- select.explore(fit, alternative = "greater")
  
  expect_s3_class(result, "select.explore")
  expect_named(result, c("pcor_mat_zero", "pcor_mat", "pcor_sd_fisher", "Adj_20", "Adj_02", "BF_20", "BF_02", "BF_cut", "method", "alternative", "call", "type", "formula", "analytic", "object"))
  expect_true(is.matrix(result$pcor_mat_zero))
  expect_true(is.matrix(result$pcor_mat))
  expect_true(is.matrix(result$Adj_20))
  expect_true(is.matrix(result$Adj_02))
})

test_that("select.explore returns expected structure for exhaustive alternative", {
  Y <- matrix(rnorm(100), ncol = 4)
  fit <- explore(Y, progress = FALSE)
  result <- select.explore(fit, alternative = "exhaustive")

  expect_s3_class(result, "select.explore")
  expect_named(result, c("post_prob", "neg_mat", "pos_mat", "null_mat", "alternative", "pcor_mat", "pcor_sd_fisher", "call", "prob", "method", "type", "formula", "analytic", "object"))
  expect_true(is.data.frame(result$post_prob))
  expect_true(is.matrix(result$neg_mat))
  expect_true(is.matrix(result$pos_mat))
  expect_true(is.matrix(result$null_mat))
})

test_that("select.explore returns expected structure for less alternative", {
  Y <- matrix(rnorm(100), ncol = 4)
  fit <- explore(Y, progress = FALSE)
  result <- select.explore(fit, alternative = "less")

  expect_s3_class(result, "select.explore")
  expect_true("Adj_20" %in% names(result))
  expect_true("Adj_02" %in% names(result))
  expect_true("BF_20" %in% names(result))
})

test_that("select.explore two.sided adjacency matrices are binary", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "two.sided")

  expect_true(all(sel$Adj_10 %in% c(0, 1)))
  expect_true(all(sel$Adj_01 %in% c(0, 1)))
})

test_that("select.explore two.sided pcor_mat is symmetric", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "two.sided")

  expect_true(isSymmetric(sel$pcor_mat))
})

test_that("select.explore exhaustive probabilities sum to 1", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "exhaustive")

  prob_sums <- sel$post_prob$prob_zero + sel$post_prob$prob_greater + sel$post_prob$prob_less
  expect_true(all(abs(prob_sums - 1) < 0.01))
})

test_that("select.explore respects BF_cut parameter", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)

  sel_3 <- select(fit, BF_cut = 3)
  sel_10 <- select(fit, BF_cut = 10)

  # Higher BF_cut should have fewer or equal selected edges
  expect_true(sum(sel_10$Adj_10) <= sum(sel_3$Adj_10))
})

test_that("select.explore print method works for two.sided", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "two.sided")

  output <- capture.output(print(sel))

  expect_true(length(output) > 0)
  expect_true(any(grepl("BGGM", output)))
  expect_true(any(grepl("two.sided", output)))
})

test_that("select.explore print method works for greater", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "greater")

  output <- capture.output(print(sel))

  expect_true(any(grepl("greater", output)))
})

test_that("select.explore print method works for less", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "less")

  output <- capture.output(print(sel))

  expect_true(any(grepl("less", output)))
})

test_that("select.explore print method works for exhaustive", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "exhaustive")

  output <- capture.output(print(sel))

  expect_true(any(grepl("exhaustive", output)))
})

test_that("summary.select.explore returns correct class", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit)
  summ <- summary(sel)

  expect_s3_class(summ, "BGGM")
  expect_s3_class(summ, "summary.select.explore")
})

test_that("summary.select.explore two.sided has correct columns", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "two.sided")
  summ <- summary(sel)

  expected_cols <- c("Relation", "Post.mean", "Post.sd.fisher", "Pr.H0", "Pr.H1")
  expect_equal(colnames(summ$summary), expected_cols)
})

test_that("summary.select.explore greater has correct columns", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "greater")
  summ <- summary(sel)

  expect_true("Pr.H0" %in% colnames(summ$summary))
  expect_true("Pr.H1" %in% colnames(summ$summary))
})

test_that("summary.select.explore exhaustive has correct columns", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "exhaustive")
  summ <- summary(sel)

  expect_true("Pr.H0" %in% colnames(summ$summary))
  expect_true("Pr.H1" %in% colnames(summ$summary))
  expect_true("Pr.H2" %in% colnames(summ$summary))
})

test_that("summary.select.explore print method works", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit)
  summ <- summary(sel)

  output <- capture.output(print(summ))

  expect_true(length(output) > 0)
  expect_true(any(grepl("BGGM", output)))
})

test_that("plot.summary.select.explore returns ggplot", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit)
  summ <- summary(sel)

  plt <- plot(summ)

  expect_true(inherits(plt, "ggplot"))
})

test_that("plot.summary.select.explore respects size parameter", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit)
  summ <- summary(sel)

  plt <- plot(summ, size = 3)

  expect_true(inherits(plt, "ggplot"))
})

test_that("plot.summary.select.explore respects color parameter", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit)
  summ <- summary(sel)

  plt <- plot(summ, color = "red")

  expect_true(inherits(plt, "ggplot"))
})

test_that("select.explore with col_names = FALSE", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit)
  summ <- summary(sel, col_names = FALSE)

  # Should use numeric indices
  expect_true(any(grepl("^[0-9]+--[0-9]+$", summ$summary$Relation)))
})

test_that("select.explore without column names", {
  set.seed(123)
  Y <- as.matrix(BGGM::bfi[1:100, 1:5])
  colnames(Y) <- NULL

  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "exhaustive")

  # Should work with numeric edge names
  expect_true(any(grepl("^[0-9]+--[0-9]+$", sel$post_prob$edge)))
})

# ---- Regression tests: pin internal consistency of BF_cut behavior ----

test_that("BF_10 and BF_01 are reciprocals for two.sided", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "two.sided")

  off_diag <- upper.tri(sel$BF_10)
  expect_true(all(abs(sel$BF_10[off_diag] * sel$BF_01[off_diag] - 1) < 1e-10))
})

test_that("Adj_10 is 1 exactly where BF_10 > BF_cut for two.sided", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  BF_cut <- 3
  sel <- select(fit, BF_cut = BF_cut, alternative = "two.sided")

  off_diag <- upper.tri(sel$BF_10)
  expect_equal(sel$Adj_10[off_diag], as.numeric(sel$BF_10[off_diag] > BF_cut))
})

test_that("Adj_01 is 1 exactly where BF_10 < 1/BF_cut for two.sided", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  BF_cut <- 3
  sel <- select(fit, BF_cut = BF_cut, alternative = "two.sided")

  off_diag <- upper.tri(sel$BF_10)
  expect_equal(sel$Adj_01[off_diag], as.numeric(sel$BF_10[off_diag] < 1 / BF_cut))
})

test_that("pcor_mat_zero is zero where Adj_10 is 0 for two.sided", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "two.sided")

  expect_true(all(sel$pcor_mat_zero[sel$Adj_10 == 0] == 0))
})

test_that("pcor_mat_zero is nonzero where Adj_10 is 1 for two.sided", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, alternative = "two.sided")

  if (any(sel$Adj_10 == 1)) {
    expect_true(all(sel$pcor_mat_zero[sel$Adj_10 == 1] != 0))
  }
})

test_that("exhaustive prob field equals BF_cut / (BF_cut + 1)", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  BF_cut <- 3
  sel <- select(fit, BF_cut = BF_cut, alternative = "exhaustive")

  expect_equal(sel$prob, BF_cut / (BF_cut + 1))
})

test_that("Adj_20 is 1 exactly where BF_20 > BF_cut for greater", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  BF_cut <- 3
  sel <- select(fit, BF_cut = BF_cut, alternative = "greater")

  off_diag <- upper.tri(sel$BF_20)
  expect_equal(sel$Adj_20[off_diag], as.numeric(sel$BF_20[off_diag] > BF_cut))
})

# ---- BMA tests: will fail until method = "BMA" is implemented ----

test_that("BMA two.sided returns correct structure", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "two.sided")

  expect_s3_class(sel, "select.explore")
  expect_true(all(c("pcor_mat_zero", "pcor_mat", "Adj_10", "Adj_01",
                    "BF_10", "BF_01", "prior.prob.H0", "method") %in% names(sel)))
  expect_true(is.na(sel$BF_cut))
  expect_equal(sel$method, "BMA")
  expect_equal(sel$prior.prob.H0, 0.5)
})

test_that("BMA two.sided Adj_10 is binary", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "two.sided")

  expect_true(all(sel$Adj_10 %in% c(0, 1)))
  expect_true(all(sel$Adj_01 %in% c(0, 1)))
})

test_that("BMA two.sided Adj_10 is consistent with pcor_mat_zero != 0", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "two.sided")

  off_diag <- upper.tri(sel$Adj_10)
  expect_equal(sel$Adj_10[off_diag], as.numeric(sel$pcor_mat_zero[off_diag] != 0))
})

test_that("BMA two.sided pcor_mat_zero is symmetric", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "two.sided")

  expect_true(isSymmetric(sel$pcor_mat_zero))
})

test_that("BMA two.sided prior.prob.H0 = 0 selects all edges", {
  set.seed(123)
  Y <- BGGM::bfi[1:500, 1:5]
  fit <- explore(Y, iter = 500, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "two.sided", prior.prob.H0 = 0)

  off_diag <- upper.tri(sel$Adj_10)
  expect_true(all(sel$Adj_10[off_diag] == 1))
})

test_that("BMA two.sided prior.prob.H0 = 1 selects no edges", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "two.sided", prior.prob.H0 = 1)

  expect_true(all(sel$Adj_10 == 0))
})

test_that("BMA two.sided stores prior.prob.H0", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "two.sided", prior.prob.H0 = 0.75)

  expect_equal(sel$prior.prob.H0, 0.75)
})

test_that("higher prior.prob.H0 selects fewer or equal edges in BMA two.sided", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)

  sel_low  <- select(fit, method = "BMA", alternative = "two.sided", prior.prob.H0 = 0.1)
  sel_high <- select(fit, method = "BMA", alternative = "two.sided", prior.prob.H0 = 0.9)

  expect_true(sum(sel_high$Adj_10) <= sum(sel_low$Adj_10))
})

test_that("BMA greater returns correct structure", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "greater")

  expect_s3_class(sel, "select.explore")
  expect_true(all(c("Adj_20", "Adj_02", "BF_20", "BF_02") %in% names(sel)))
  expect_equal(sel$method, "BMA")
})

test_that("BMA greater pcor_mat_zero has no negative values", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "greater")

  expect_true(all(sel$pcor_mat_zero >= 0))
})

test_that("BMA less returns correct structure", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "less")

  expect_s3_class(sel, "select.explore")
  expect_true(all(c("Adj_20", "Adj_02", "BF_20", "BF_02") %in% names(sel)))
  expect_equal(sel$method, "BMA")
})

test_that("BMA less pcor_mat_zero has no positive values", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "less")

  expect_true(all(sel$pcor_mat_zero <= 0))
})

test_that("BMA exhaustive stops with informative error", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)

  expect_error(
    select(fit, method = "BMA", alternative = "exhaustive"),
    "exhaustive"
  )
})

test_that("BMA print method works without error for two.sided", {
  set.seed(123)
  Y <- BGGM::bfi[1:100, 1:5]
  fit <- explore(Y, iter = 100, progress = FALSE)
  sel <- select(fit, method = "BMA", alternative = "two.sided")

  output <- capture.output(print(sel))
  expect_true(length(output) > 0)
  expect_true(any(grepl("BGGM", output)))
  expect_false(any(grepl("Bayes Factor: NA", output)))
})
