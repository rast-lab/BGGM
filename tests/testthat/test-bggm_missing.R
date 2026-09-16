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

test_that("bggm_missing pools the post-burn-in draws of all imputations", {
    set.seed(123)
    Y <- matrix(rnorm(200), ncol = 5)
    Y[sample(length(Y), 30)] <- NA
    imp <- mice::mice(Y, m = 3, print = FALSE)

    for (method in c("explore", "estimate")) {
        res <- bggm_missing(imp, method = method, iter = 100,
                            progress = FALSE, seed = 1)
        # explore stores no burn-in draws, estimate stores 50
        n_burn <- if (method == "explore") 0 else 50
        idx    <- n_burn + 1:300

        expect_equal(res$iter, 300)
        expect_equal(BGGM:::post_draw_idx(res), idx)
        expect_equal(dim(res$post_samp$pcors)[3], n_burn + 300)
        expect_equal(dim(res$post_samp$fisher_z)[3], n_burn + 300)

        # pooled draws are the post-burn-in draws of the separate fits
        dat <- mice::complete(imp, action = "long")
        fits <- lapply(1:3, function(i) {
            Yi <- as.matrix(dat[dat$.imp == i, !(names(dat) %in% c(".imp", ".id"))])
            if (method == "explore") explore(Yi, iter = 100, progress = FALSE, seed = 1)
            else estimate(Yi, iter = 100, progress = FALSE, seed = 1)
        })
        first <- BGGM:::post_draw_idx(fits[[1]])
        expect_equal(res$post_samp$pcors[, , idx[1:100]], fits[[1]]$post_samp$pcors[, , first])
        expect_equal(res$post_samp$pcors[, , idx[201:300]], fits[[3]]$post_samp$pcors[, , first])

        # summaries use all pooled draws
        expect_equal(res$pcor_mat, apply(res$post_samp$pcors[, , idx], 1:2, mean))
        if (method == "explore") {
            expect_equal(res$post_samp$z_mean,
                         apply(res$post_samp$fisher_z[, , idx], 1:2, mean))
            expect_s3_class(select(res), "select.explore")
        }
    }
})

test_that("bggm_missing explore ignores store_post_draws = FALSE with a warning", {
    set.seed(123)
    Y <- matrix(rnorm(100), ncol = 5)
    Y[sample(length(Y), 20)] <- NA
    imp <- mice::mice(Y, m = 2, print = FALSE)
    expect_warning(
        res <- bggm_missing(imp, method = "explore", iter = 100,
                            progress = FALSE, store_post_draws = FALSE),
        "ignored by bggm_missing"
    )
    expect_false(is.null(res$post_samp$pcors))
})
