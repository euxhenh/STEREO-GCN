import math
from collections import namedtuple
from itertools import zip_longest
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch
from scipy.stats import norm
from sklearn.utils import check_consistent_length

from .graph_ops import adj_from_edge_index

CMAP = sns.color_palette("viridis", as_cmap=True).copy()
CMAP.set_bad("black")

PALETTE = [
    "#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F",
    "#B3B3B3", "#E5C494", "#9C9DBB", "#E6946B", "#DA8DC4", "#AFCC63",
    "#F2D834", "#E8C785", "#BAB5AE", "#D19C75", "#AC9AAD", "#CD90C5",
    "#B8C173", "#E5D839", "#ECCA77", "#C1B7AA", "#BBA37E", "#BC979D",
    "#C093C6", "#C1B683", "#D8D83E", "#F0CD68", "#C8BAA5", "#A6AB88",
    "#CC958F", "#B396C7", "#CBAB93", "#CCD844", "#F3D05A", "#CFBCA1",
    "#90B291", "#DC9280", "#A699C8", "#D4A0A3", "#BFD849", "#F7D34B",
    "#D6BF9C", "#7BBA9B", "#EC8F71", "#999CC9", "#DD95B3", "#B2D84E",
    "#FBD63D", "#DDC198"
]


def get_col_i(pal, i):
    return pal[i % len(pal)]


def grid_plot(
    xlist,
    fn,
    fn_kwargs=None,
    titlelist=None,
    ncols=5,
    sharex=True,
    sharey=True,
    seaborn_fn=True,
    figsize=None,
    savepath=None,
    logger=None,
    logger_title=None,
):
    """Grid plot.

    xlist: List of data
        Each element will be passed to fn.
    fn: callable
        The function to execute.
    fn_kwargs: dict
        Named arguments to pass to fn call.
    ncols: int
        Number of columns in the grid.
    """
    fn_kwargs = fn_kwargs or {}
    nrows = math.ceil(len(xlist) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize or (ncols*5, nrows*5),
        sharex=sharex,
        sharey=sharey,
    )

    for i, (x, ax) in enumerate(zip_longest(xlist, axes.flat, fillvalue=None)):
        if x is None:
            ax.remove()
            continue

        if seaborn_fn:
            fn_kwargs['ax'] = ax

        fn(x, **fn_kwargs)
        if titlelist is not None:
            ax.set_title(titlelist[i])

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    if logger and logger_title:
        logger.log({logger_title: plt})

    plt.show()


def binarize(A, q=0.95, min_only=False):
    m = np.quantile(A.flat, q)
    A = np.where(A >= m, 1 if not min_only else A, 0)
    return A


def compare_graphs(n_nodes, indices, w_true, w_pred, q=0.95, genes=None):
    """Compares two graphs specified by indices and weights.
    """
    A_init = adj_from_edge_index(n_nodes, indices)
    A_true = adj_from_edge_index(n_nodes, indices, w_true)
    A_pred = adj_from_edge_index(n_nodes, indices, w_pred)
    A_true_bi = binarize(A_true, q=q)
    A_pred_bi = binarize(A_pred, q=q)
    A_diff = np.abs(A_true_bi - A_pred_bi)
    A_olap_bi = A_true_bi * A_pred_bi
    A_olap = A_true * A_pred_bi

    d = {
        'A_init': A_init,
        'A_true': A_true,
        'A_true_bi': A_true_bi,
        'A_olap_bi': A_olap_bi,
        'A_diff': A_diff,
        'A_pred': A_pred,
        'A_pred_bi': A_pred_bi,
        'A_olap': A_olap,
    }

    titlelist = [
        'DoF',
        'Ground Truth',
        f'Ground Truth {q=}',
        'Olap (true q * pred q)',
        f'Difference {q=}',
        'Prediction',
        f'Prediction {q=}',
        'Olap (true * pred q)',
    ]

    assert len(d) == len(titlelist)

    grid_plot(
        list(d.values()),
        ncols=4,
        fn=sns.heatmap,
        fn_kwargs=dict(
            square=True,
            cmap=CMAP,
            mask=A_init == 0,
        ),
        titlelist=titlelist,
    )

    if genes is not None:
        olap = np.intersect1d(
            A_true.sum(axis=1).argsort()[-5:],
            A_pred.sum(axis=1).argsort()[-5:],
        )
        print("Source overlap")
        print(list(zip(olap, genes[olap])))
        olap = np.intersect1d(
            A_true.sum(axis=0).argsort()[-5:],
            A_pred.sum(axis=0).argsort()[-5:],
        )
        print("Target overlap")
        print(list(zip(olap, genes[olap])))

    return d


Stat = namedtuple('Stat', ['n_selected', 'mean', 'n_pos', 'pval'])


def topq_significance(
    w_true: torch.Tensor | np.ndarray,
    w_preds: List[torch.Tensor | np.ndarray],
    q: float = 0.95,
    alpha: float = 0.05,
):
    """Plots a distribution of the mean of k randomly drawn points from
    w_true, and computes the survival function of the actual (true) sums of
    the top k points in each w_pred in w_preds. Here, k is determined by
    the quantile q.
    """
    check_consistent_length(w_true, *w_preds)
    mu, std = w_true.mean(), w_true.std()

    fig, ax = plt.subplots()
    max_mean = 0

    stats = []
    means = []

    for i, w_pred in enumerate(w_preds):
        selected_idx = np.argwhere(w_pred >= np.quantile(w_pred, q)).flatten()
        selected_mean = w_true[selected_idx].mean()
        max_mean = max(max_mean, selected_mean)
        # E[1/k * \sum x_i] = E[x]
        # std[1/k * \sum x_i] = 1/sqrt(k) std[x]
        pval = norm.sf(selected_mean, loc=mu, scale=std / len(selected_idx)**(1/2))
        ax.axvline(x=selected_mean, label=f'p{i}={pval:.5f}', color=PALETTE[i % len(PALETTE)])

        means.append(selected_mean)

        stats.append(Stat(
            len(selected_idx),
            selected_mean,
            (w_true[selected_idx] > 0).sum(),
            pval,
        ))

    # Plot normal curve
    std_approx = std / (len(w_true) * (1 - q))**(1/2)
    z = norm.ppf(1 - alpha, mu, std_approx)
    x_axis = np.linspace(min(np.min(means)-0.05, z - 0.03),
                         max(np.max(means)+0.05, z + 0.03), 100)
    ax.plot(x_axis, norm.pdf(x_axis, mu, std_approx))

    ax.axvline(z, color='black', linestyle='--', label=f'{alpha=}')

    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left')

    plt.show()

    return stats


def draw_graph(
    A: np.ndarray,
    genes: List[str] = None,
    ax: plt.Axes = None,
    figsize: Tuple[int, int] = (15, 15),
    logger=None,
    logger_title: str | None = None,
):
    """Draws graph given by adjacency matrix A.
    """
    evolution_mode = False
    if A.min() < 0:
        evolution_mode = True
    if genes is None:
        genes = np.arange(A.shape[0])

    check_consistent_length(A, genes)

    g = nx.DiGraph()
    g.add_nodes_from(genes)

    src, trg = A.nonzero()
    outgoing_sum = np.abs(A).sum(axis=1)
    nodesize = outgoing_sum * 2000 / outgoing_sum.max()
    edgewidth = A[src, trg]
    src_gene, trg_gene = genes[src], genes[trg]
    gene_to_idx = {g: idx for idx, g in enumerate(genes)}

    for s, t, w in zip(src_gene, trg_gene, edgewidth):
        if not evolution_mode:
            color = get_col_i(PALETTE, gene_to_idx[s])
        else:
            color = "#36AE7C" if w > 0 else "#FFA8A8"
        g.add_edge(s, t, color=color)
    edge_color = [g[u][v]['color'] for u, v in g.edges()]
    node_color = [get_col_i(PALETTE, idx) for idx in gene_to_idx.values()]

    pos = nx.circular_layout(g)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_edges(
        g,
        pos,
        alpha=0.8,
        width=np.abs(edgewidth) * 3,
        node_size=nodesize,
        arrowsize=10,
        edge_color=edge_color,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        g,
        pos,
        node_size=nodesize,
        node_color=node_color,
        alpha=0.9,
        ax=ax,
    )
    label_options = {"ec": "k", "fc": "white", "alpha": 0.7}
    nx.draw_networkx_labels(
        g,
        pos,
        font_size=14,
        bbox=label_options,
        ax=ax,
    )

    sns.despine(ax=ax, bottom=True, left=True)

    if logger and logger_title:
        logger.log({logger_title: plt})


def interpolate_2dimage(im, colorscale='vlag', reverse=False):
    """Maps a 2D image to a 3D image using the given colormap."""
    is_pt = isinstance(im, torch.Tensor)
    lib = torch if is_pt else np
    assert im.ndim == 2
    colors = sns.color_palette(colorscale)
    if reverse:
        colors = list(reversed(colors))

    # First we bin each value in im so that we know which colors
    # to use for interpolation
    unit = (im - im.min()) / (im.max() - im.min())
    binned = (unit * (len(colors) - 1)) - 0.5
    binned = lib.round(binned)
    if is_pt:
        binned = binned.long().clip(min=0)
    else:
        binned = binned.astype(int).clip(min=0)
    # empty 3d im
    im3d = lib.zeros((*im.shape, 3))
    if is_pt:
        im3d = im3d.to(im.device)
    for i in range(len(colors) - 1):  # Small for loop so it's ok
        r1, g1, b1 = colors[i]
        r2, g2, b2 = colors[i + 1]
        idx = lib.where(binned == i)
        if is_pt and idx[0].numel() == 0 or not is_pt and idx[0].size == 0:
            continue
        minv, maxv = im[idx].min(), im[idx].max()
        if minv == maxv:
            intermeds = 0.5 if not is_pt else torch.Tensor([0.5]).to(im.device)
        else:
            intermeds = (im[idx] - minv) / (maxv - minv)
        im3d[idx] = lib.stack([
            (r2 - r1) * intermeds + r1,
            (g2 - g1) * intermeds + g1,
            (b2 - b1) * intermeds + b1
        ], axis=-1)
    return im3d
