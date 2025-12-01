# BGGM: Bayesian Gaussian Graphical Models

The `R` package **BGGM** provides tools for making Bayesian inference in
Gaussian graphical models (GGM). The methods are organized around two
general approaches for Bayesian inference: (1) estimation (Williams
2018) and (2) hypothesis testing (Williams and Mulder 2019) . The key
distinction is that the former focuses on either the posterior or
posterior predictive distribution, whereas the latter focuses on model
comparison with the Bayes factor.

The methods in **BGGM** build upon existing algorithms that are
well-known in the literature. The central contribution of **BGGM** is to
extend those approaches:

1.  Bayesian estimation with the novel matrix-F prior distribution
    (Mulder and Pericchi 2018) .

    - Estimation
      [`estimate`](https://rast-lab.github.io/BGGM/reference/estimate.md).

2.  Bayesian hypothesis testing with the novel matrix-F prior
    distribution (Mulder and Pericchi 2018) .

    - Exploratory hypothesis testing
      [`explore`](https://rast-lab.github.io/BGGM/reference/explore.md).

    - Confirmatory hypothesis testing
      [`confirm`](https://rast-lab.github.io/BGGM/reference/confirm.md).

3.  Comparing GGMs (Williams et al. 2020)

    - Partial correlation differences
      [`ggm_compare_estimate`](https://rast-lab.github.io/BGGM/reference/ggm_compare_estimate.md).

    - Posterior predictive check
      [`ggm_compare_ppc`](https://rast-lab.github.io/BGGM/reference/ggm_compare_ppc.md).

    - Exploratory hypothesis testing
      [`ggm_compare_explore`](https://rast-lab.github.io/BGGM/reference/ggm_compare_explore.md).

    - Confirmatory hypothesis testing
      [`ggm_compare_confirm`](https://rast-lab.github.io/BGGM/reference/ggm_compare_confirm.md).

4.  Extending inference beyond the conditional (in)dependence structure

    - Predictability with Bayesian variance explained (Gelman et
      al. 2019)
      [`predictability`](https://rast-lab.github.io/BGGM/reference/predictability.md).

    - Posterior uncertainty in the partial correlations
      [`estimate`](https://rast-lab.github.io/BGGM/reference/estimate.md).

    - Custom Network Statistics
      [`roll_your_own`](https://rast-lab.github.io/BGGM/reference/roll_your_own.md).

Furthermore, the computationally intensive tasks are written in `c++`
via the `R` package **Rcpp** (Eddelbuettel et al. 2011) and the `c++`
library **Armadillo** (Sanderson and Curtin 2016) , there are plotting
functions for each method, control variables can be included in the
model, and there is support for missing values
[`bggm_missing`](https://rast-lab.github.io/BGGM/reference/bggm_missing.md).

**Supported Data Types**:

- Continuous: The continuous method was described in Williams and
  Mulder (2019) .

- Binary: The binary method builds directly upon in Talhouk et
  al. (2012) , that, in turn, built upon the approaches of Lawrence et
  al. (2008) and Webb and Forster (2008) (to name a few).

- Ordinal: Ordinal data requires sampling thresholds. There are two
  approach included in **BGGM**: (1) the customary approach described in
  in Albert and Chib (1993) (the default) and the 'Cowles' algorithm
  described in in Cowles (1996) .

- Mixed: The mixed data (a combination of discrete and continuous)
  method was introduced in Hoff (2007) . This is a semi-parametric
  copula model (i.e., a copula GGM) based on the ranked likelihood. Note
  that this can be used for data consisting entirely of ordinal data.

**Additional Features**:

The primary focus of `BGGM` is Gaussian graphical modeling (the inverse
covariance matrix). The residue is a suite of useful methods not
explicitly for GGMs:

1.  Bivariate correlations for binary (tetrachoric), ordinal
    (polychoric), mixed (rank based), and continuous (Pearson's) data
    [`zero_order_cors`](https://rast-lab.github.io/BGGM/reference/zero_order_cors.md).

2.  Multivariate regression for binary (probit), ordinal (probit), mixed
    (rank likelihood), and continous data
    ([`estimate`](https://rast-lab.github.io/BGGM/reference/estimate.md)).

3.  Multiple regression for binary (probit), ordinal (probit), mixed
    (rank likelihood), and continuous data (e.g.,
    [`coef.estimate`](https://rast-lab.github.io/BGGM/reference/coef.estimate.md)).

**Note on Conditional (In)dependence Models for Latent Data**:

All of the data types (besides continuous) model latent data. That is,
unobserved (latent) data is assumed to be Gaussian. For example, a
tetrachoric correlation (binary data) is a special case of a polychoric
correlation (ordinal data). Both capture relations between "theorized
normally distributed continuous **latent** variables"
([Wikipedia](https://en.wikipedia.org/wiki/Polychoric_correlation)). In
both instances, the corresponding partial correlation between observed
variables is conditioned on the remaining variables in the *latent*
space. This implies that interpretation is similar to continuous data,
but with respect to latent variables. We refer interested users to page
2364, section 2.2, in Webb and Forster (2008) .

**High Dimensional Data?**

**BGGM** was built specifically for social-behavioral scientists. Of
course, the methods can be used by all researchers. However, there is
currently *not* support for high-dimensional data (i.e., more variables
than observations) that are common place in the genetics literature.
These data are rare in the social-behavioral sciences. In the future,
support for high-dimensional data may be added to **BGGM**.

## References

Albert JH, Chib S (1993). “Bayesian analysis of binary and polychotomous
response data.” *Journal of the American statistical Association*,
**88**(422), 669–679.  
  
Cowles MK (1996). “Accelerating Monte Carlo Markov chain convergence for
cumulative-link generalized linear models.” *Statistics and Computing*,
**6**(2), 101–111.
[doi:10.1007/bf00162520](https://doi.org/10.1007/bf00162520) .  
  
Eddelbuettel D, François R, Allaire J, Ushey K, Kou Q, Russel N,
Chambers J, Bates D (2011). “Rcpp: Seamless R and C++ integration.”
*Journal of Statistical Software*, **40**(8), 1–18.  
  
Gelman A, Goodrich B, Gabry J, Vehtari A (2019). “R-squared for Bayesian
Regression Models.” *American Statistician*, **73**(3), 307–309. ISSN
15372731.  
  
Hoff PD (2007). “Extending the rank likelihood for semiparametric copula
estimation.” *The Annals of Applied Statistics*, **1**(1), 265–283.
[doi:10.1214/07-AOAS107](https://doi.org/10.1214/07-AOAS107) .  
  
Lawrence E, Bingham D, Liu C, Nair VN (2008). “Bayesian inference for
multivariate ordinal data using parameter expansion.” *Technometrics*,
**50**(2), 182–191.  
  
Mulder J, Pericchi L (2018). “The Matrix-F Prior for Estimating and
Testing Covariance Matrices.” *Bayesian Analysis*, 1–22. ISSN 19316690,
[doi:10.1214/17-BA1092](https://doi.org/10.1214/17-BA1092) .  
  
Sanderson C, Curtin R (2016). “Armadillo: a template-based C++ library
for linear algebra.” *Journal of Open Source Software*, **1**(2), 26.
[doi:10.21105/joss.00026](https://doi.org/10.21105/joss.00026) .  
  
Talhouk A, Doucet A, Murphy K (2012). “Efficient Bayesian inference for
multivariate probit models with sparse inverse correlation matrices.”
*Journal of Computational and Graphical Statistics*, **21**(3),
739–757.  
  
Webb EL, Forster JJ (2008). “Bayesian model determination for
multivariate ordinal and binary data.” *Computational statistics & data
analysis*, **52**(5), 2632–2649.
[doi:10.1016/j.csda.2007.09.008](https://doi.org/10.1016/j.csda.2007.09.008)
.  
  
Williams DR (2018). “Bayesian Estimation for Gaussian Graphical Models:
Structure Learning, Predictability, and Network Comparisons.” *arXiv*.
[doi:10.31234/OSF.IO/X8DPR](https://doi.org/10.31234/OSF.IO/X8DPR) .  
  
Williams DR, Mulder J (2019). “Bayesian Hypothesis Testing for Gaussian
Graphical Models: Conditional Independence and Order Constraints.”
*PsyArXiv*.
[doi:10.31234/osf.io/ypxd8](https://doi.org/10.31234/osf.io/ypxd8) .  
  
Williams DR, Rast P, Pericchi LR, Mulder J (2020). “Comparing Gaussian
graphical models with the posterior predictive distribution and Bayesian
model selection.” *Psychological Methods*.
[doi:10.1037/met0000254](https://doi.org/10.1037/met0000254) .
