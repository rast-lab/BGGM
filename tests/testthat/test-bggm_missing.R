library(testthat)
library(BGGM)

test_that("bggm_missing handles missing data with method 'estimate'", {
    set.seed(123)
    
    # Create a sample dataset with missing values
    Y <- matrix(rnorm(100), ncol = 5)
    Y[sample(length(Y), 20)] <- NA
    
    # Impute missing data
    imp <- mice::mice(Y, m = 5, print = FALSE)
    
    # Test bggm_missing with method 'estimate'
    result <- bggm_missing(imp, method = "estimate", iter = 250, progress = FALSE)
    
    expect_s3_class(result, "estimate")
    expect_true("post_samp" %in% names(result))
    expect_true("pcors" %in% names(result$post_samp))
})

test_that("bggm_missing handles missing data with method 'explore'", {
    set.seed(123)
    
    # Create a sample dataset with missing values
    Y <- matrix(rnorm(100), ncol = 5)
    Y[sample(length(Y), 20)] <- NA
    
    # Impute missing data
    imp <- mice::mice(Y, m = 5, print = FALSE)
    
    # Test bggm_missing with method 'explore'
    result <- bggm_missing(imp, method = "explore", iter = 250, progress = FALSE)
    
    expect_s3_class(result, "explore")
    expect_true("post_samp" %in% names(result))
    expect_true("pcors" %in% names(result$post_samp))
})

test_that("bggm_missing drops .id by name, not position (mice >= 3.17.0, issue #2)", {
    # Regression test: complete(action = "long") places .imp/.id in the LAST
    # two columns since mice 3.17.0. Removing column 2 by position therefore
    # dropped a real data column and let .id leak into the model. Named columns
    # make the leak observable: the analyzed data (res$Y) must keep all five
    # variables and must not contain ".id".
    set.seed(123)
    Y <- matrix(rnorm(200), ncol = 5)
    Y[sample(length(Y), 30)] <- NA
    colnames(Y) <- c("A", "B", "C", "D", "E")
    Y <- as.data.frame(Y)

    imp <- mice::mice(Y, m = 3, print = FALSE)
    result <- bggm_missing(imp, method = "estimate", iter = 250, progress = FALSE)

    expect_true(all(c("A", "B", "C", "D", "E") %in% colnames(result$Y)))
    expect_false(".id" %in% colnames(result$Y))
    expect_equal(ncol(result$Y), 5)
})
