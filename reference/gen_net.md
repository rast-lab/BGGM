# Simulate a Partial Correlation Matrix

Simulate a Partial Correlation Matrix

## Usage

``` r
gen_net(p = 20, edge_prob = 0.3, lb = 0.05, ub = 0.3)
```

## Arguments

- p:

  number of variables (nodes)

- edge_prob:

  connectivity

- lb:

  lower bound for the partial correlations

- ub:

  upper bound for the partial correlations

## Value

A list containing the following:

- **pcor**: Partial correlation matrix, encoding the conditional
  (in)dependence structure.

- **cors**: Correlation matrix.

- **adj**: Adjacency matrix.

- **trys**: Number of attempts to obtain a positive definite matrix.

## Note

The function checks for a valid matrix (positive definite), but
sometimes this will still fail. For example, for larger `p`, to have
large partial correlations this requires a sparse GGM (accomplished by
setting `edge_prob` to a small value).

## Examples

``` r
true_net <- gen_net(p = 10)
```
