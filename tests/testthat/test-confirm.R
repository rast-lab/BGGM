library(testthat)
library(BGGM)

# Test data setup
set.seed(123)
test_data <- matrix(rnorm(150), ncol = 5)
colnames(test_data) <- paste0("V", 1:5)

# Basic functionality tests
test_that("confirm works with simple equality hypothesis", {
  # Hypothesis: partial correlation between V1 and V2 equals 0
  hypothesis <- "V1--V2 = 0"

  result <- confirm(
    Y = test_data,
    hypothesis = hypothesis,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, c("BGGM", "confirm"))
  expect_true(is.list(result))
})

test_that("confirm returns correct structure", {
  hypothesis <- "V1--V2 = 0"

  result <- confirm(
    Y = test_data,
    hypothesis = hypothesis,
    iter = 100,
    progress = FALSE
  )

  expect_true("hypothesis" %in% names(result))
})

test_that("confirm works with inequality hypothesis (greater)", {
  # Hypothesis: partial correlation between V1 and V2 is greater than 0
  hypothesis <- "V1--V2 > 0"

  result <- confirm(
    Y = test_data,
    hypothesis = hypothesis,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "confirm")
})

test_that("confirm works with inequality hypothesis (less)", {
  # Hypothesis: partial correlation between V1 and V2 is less than 0
  hypothesis <- "V1--V2 < 0"

  result <- confirm(
    Y = test_data,
    hypothesis = hypothesis,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "confirm")
})

test_that("confirm works with multiple edges in hypothesis", {
  # Hypothesis: V1--V2 equals V3--V4
  hypothesis <- "V1--V2 = V3--V4"

  result <- confirm(
    Y = test_data,
    hypothesis = hypothesis,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "confirm")
})

test_that("confirm works with compound hypotheses using semicolons", {
  # Multiple hypotheses separated by semicolons
  hypothesis <- "V1--V2 > 0; V3--V4 > 0"

  result <- confirm(
    Y = test_data,
    hypothesis = hypothesis,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "confirm")
})

# Test different data types
test_that("confirm handles binary data", {
  set.seed(456)
  binary_data <- matrix(as.integer(rnorm(150) > 0), ncol = 5)
  colnames(binary_data) <- paste0("V", 1:5)

  hypothesis <- "V1--V2 = 0"

  result <- confirm(
    Y = binary_data,
    hypothesis = hypothesis,
    type = "binary",
    iter = 50,
    progress = FALSE
  )

  expect_s3_class(result, "confirm")
  expect_identical(result$type, "binary")
})

test_that("confirm handles ordinal data", {
  set.seed(789)
  ordinal_data <- matrix(sample(1:4, 150, replace = TRUE), ncol = 5)
  colnames(ordinal_data) <- paste0("V", 1:5)

  hypothesis <- "V1--V2 = 0"

  result <- confirm(
    Y = ordinal_data,
    hypothesis = hypothesis,
    type = "ordinal",
    iter = 50,
    progress = FALSE
  )

  expect_s3_class(result, "confirm")
  expect_identical(result$type, "ordinal")
})

# Test with prior estimate
test_that("confirm works with prior_sd parameter", {
  hypothesis <- "V1--V2 = 0"

  result <- confirm(
    Y = test_data,
    hypothesis = hypothesis,
    prior_sd = 0.5,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "confirm")
})

# Test prior_sd parameter
test_that("confirm accepts custom prior_sd", {
  hypothesis <- "V1--V2 = 0"

  result <- confirm(
    Y = test_data,
    hypothesis = hypothesis,
    prior_sd = 0.25,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "confirm")
})

# Test print method works (confirm uses print, not summary)
test_that("print.confirm works", {
  hypothesis <- "V1--V2 = 0"

  fit <- confirm(
    Y = test_data,
    hypothesis = hypothesis,
    iter = 100,
    progress = FALSE
  )

  # print should work without error
  expect_output(print(fit))
})

# Test edge cases
test_that("confirm handles data with column names containing numbers", {
  set.seed(111)
  data_named <- matrix(rnorm(100), ncol = 5)
  colnames(data_named) <- c("item1", "item2", "item3", "item4", "item5")

  hypothesis <- "item1--item2 = 0"

  result <- confirm(
    Y = data_named,
    hypothesis = hypothesis,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "confirm")
})

# Test BF_01 and BF_10 are computed
test_that("confirm returns Bayes factors", {
  hypothesis <- "V1--V2 = 0"

  result <- confirm(
    Y = test_data,
    hypothesis = hypothesis,
    iter = 100,
    progress = FALSE
  )

  # Should have BF information
  expect_true(any(c("BF", "BF_01", "BF_10", "info") %in% names(result)))
})

# Test hypothesis with zero comparison
test_that("confirm works with zero equality constraint", {
  hypothesis <- "V1--V2 = 0"

  result <- confirm(
    Y = test_data,
    hypothesis = hypothesis,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "confirm")
})

# Test with larger dataset
test_that("confirm works with larger sample size", {
  set.seed(222)
  large_data <- matrix(rnorm(500), ncol = 5)
  colnames(large_data) <- paste0("V", 1:5)

  hypothesis <- "V1--V2 = 0"

  result <- confirm(
    Y = large_data,
    hypothesis = hypothesis,
    iter = 100,
    progress = FALSE
  )

  expect_s3_class(result, "confirm")
})
