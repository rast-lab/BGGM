#!/usr/bin/env Rscript

#' Simulate ordinal data from a latent Gaussian graphical model
#'
#' This script can be run with `Rscript` or sourced from an interactive session.
#' It generates ordinal data by sampling correlated latent Gaussian variables and
#' discretising them according to user supplied thresholds. The defaults mimic
#' the settings used when developing BGGM's ordinal sampler, but you can tweak
#' the arguments to explore other scenarios.
#'
#' Usage from the command line:
#'
#'   Rscript simulate_ordinal_data.R --n=400 --p=5 --rho=0.4 \
#'       --thresholds=-1,0,1 --seed=2024 --output=ordinal_example.csv
#'
#' If `--output` is omitted the simulated data are printed to standard output
#' (CSV format). When sourcing the file the helper function
#' `simulate_ordinal_data()` is exported into your workspace.

suppressWarnings(suppressMessages({
  if (!requireNamespace("MASS", quietly = TRUE)) {
    stop("The 'MASS' package is required. Install it with install.packages('MASS').",
         call. = FALSE)
  }
}))

# Build an AR(1) correlation matrix with parameter rho
make_ar1_cov <- function(p, rho) {
  if (p < 1) {
    stop("'p' must be >= 1", call. = FALSE)
  }
  if (!is.finite(rho) || abs(rho) >= 1) {
    stop("'rho' must be finite and satisfy |rho| < 1 for a valid AR(1) correlation.",
         call. = FALSE)
  }
  mat <- outer(seq_len(p), seq_len(p), function(i, j) rho^abs(i - j))
  diag(mat) <- 1
  mat
}

# Simulate ordinal data with identical thresholds for each variable
simulate_ordinal_data <- function(n = 400L,
                                  p = 5L,
                                  rho = 0.4,
                                  thresholds = c(-1, 0, 1),
                                  seed = NULL) {
  if (!is.null(seed)) {
    set.seed(as.integer(seed))
  }
  if (length(thresholds) < 1) {
    stop("Provide at least one interior threshold to define ordinal categories.",
         call. = FALSE)
  }
  thresholds <- sort(as.numeric(thresholds))
  Sigma <- make_ar1_cov(p, rho)
  latent <- MASS::mvrnorm(n = n, mu = rep(0, p), Sigma = Sigma)
  breaks <- c(-Inf, thresholds, Inf)
  ordinal <- apply(latent, 2, function(col) {
    cut(col, breaks = breaks, labels = FALSE)
  })
  colnames(ordinal) <- sprintf("Y%02d", seq_len(p))
  as.data.frame(ordinal)
}

write_data <- function(dat, path = NULL) {
  if (is.null(path) || identical(path, "")) {
    write.table(dat, sep = ",", row.names = FALSE, col.names = TRUE,
                quote = FALSE, file = stdout())
  } else {
    utils::write.csv(dat, file = path, row.names = FALSE)
    message("Wrote ", nrow(dat), " x ", ncol(dat), " ordinal data set to '", path, "'.")
  }
}

parse_args <- function(args) {
  defaults <- list(n = 400L, p = 5L, rho = 0.4, thresholds = c(-1, 0, 1),
                   seed = NULL, output = NULL)
  for (arg in args) {
    if (!startsWith(arg, "--")) {
      next
    }
    key_value <- sub("^--", "", arg, fixed = FALSE)
    parts <- strsplit(key_value, "=", fixed = TRUE)[[1]]
    key <- parts[1]
    value <- if (length(parts) == 2) parts[2] else ""
    switch(key,
           n = defaults$n <- as.integer(value),
           p = defaults$p <- as.integer(value),
           rho = defaults$rho <- as.numeric(value),
           thresholds = {
             pieces <- strsplit(value, ",", fixed = TRUE)[[1]]
             defaults$thresholds <- as.numeric(pieces[pieces != ""])
           },
           seed = defaults$seed <- as.numeric(value),
           output = defaults$output <- value,
           help = {
             cat("Usage: Rscript simulate_ordinal_data.R [options]\n\n",
                 "Options:\n",
                 "  --n=<int>            Number of observations (default 400)\n",
                 "  --p=<int>            Number of variables (default 5)\n",
                 "  --rho=<double>       AR(1) correlation parameter (default 0.4)\n",
                 "  --thresholds=a,b,... Interior thresholds shared by all variables\n",
                 "                        (default -1,0,1 yielding four categories)\n",
                 "  --seed=<int>         Optional RNG seed\n",
                 "  --output=<file>      Path to write CSV output\n",
                 "  --help               Show this message\n",
                 sep = "")
             quit(status = 0)
           },
           {
             warning("Ignoring unknown option '--", key, "'.", call. = FALSE)
           }
    )
  }
  defaults
}

# Only execute when called via Rscript --------------------------------------
if (!interactive()) {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  dat <- simulate_ordinal_data(n = opts$n,
                               p = opts$p,
                               rho = opts$rho,
                               thresholds = opts$thresholds,
                               seed = opts$seed)
  write_data(dat, opts$output)
}

