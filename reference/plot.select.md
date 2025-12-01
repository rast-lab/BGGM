# Network Plot for `select` Objects

Visualize the conditional (in)dependence structure.

## Usage

``` r
# S3 method for class 'select'
plot(
  x,
  layout = "circle",
  pos_col = "#009E73",
  neg_col = "#D55E00",
  node_size = 10,
  edge_magnify = 1,
  groups = NULL,
  palette = "Set3",
  ...
)
```

## Arguments

- x:

  An object of class `select`.

- layout:

  Character string. Which graph layout (defaults is `circle`) ? See
  [gplot.layout](https://rdrr.io/pkg/sna/man/gplot.layout.html).

- pos_col:

  Character string. Color for the positive edges (defaults to `green`).

- neg_col:

  Character string. Color for the negative edges (defaults to `green`).

- node_size:

  Numeric. The size of the nodes (defaults to `10`).

- edge_magnify:

  Numeric. A value that is multiplied by the edge weights. This
  increases (\> 1) or decrease (\< 1) the line widths (defaults to 1).

- groups:

  A character string of length *p* (the number of nodes in the model).
  This indicates groups of nodes that should be the same color (e.g.,
  "clusters" or "communities").

- palette:

  A character string sepcifying the palette for the `groups`. (default
  is `Set3`). See [palette options
  here](http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/).

- ...:

  Additional options passed to
  [ggnet2](https://ggobi.github.io/ggally/reference/ggnet2.html)

## Value

An object (or list of objects) of class `ggplot` that can then be
further customized.

## Note

A more extensive example of a custom plot is provided
[here](https://donaldrwilliams.github.io/BGGM/articles/netplot.html)

## Examples

``` r
# \donttest{
#########################
### example 1: one ggm ##
#########################

# data
Y <- bfi[,1:25]

# estimate
fit <- estimate(Y, iter = 250,
                progress = FALSE)

# "communities"
comm <- substring(colnames(Y), 1, 1)

# edge set
E <- select(fit)

# plot edge set
plt_E <- plot(E, edge_magnify = 5,
              palette = "Set1",
              groups = comm)


#############################
### example 2: ggm compare ##
#############################
# compare males vs. females

# data
Y <- bfi[,1:26]

Ym <- subset(Y, gender == 1,
             select = -gender)

Yf <- subset(Y, gender == 2,
              select = -gender)

# estimate
fit <- ggm_compare_estimate(Ym, Yf, iter = 250,
                            progress = FALSE)

# "communities"
comm <- substring(colnames(Ym), 1, 1)

# edge set
E <- select(fit)

# plot edge set
plt_E <- plot(E, edge_magnify = 5,
              palette = "Set1",
              groups = comm)


# }
```
