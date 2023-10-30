import json

import numpy as np

import stereo


def load_run_ids(path: str, key: str | None = None):
    """Loads json with seed: runID pairs.
    """
    with open(path, 'r') as f:
        runs = json.load(f)
        if key is not None:
            runs = runs[key]
        runs = {int(k): v for k, v in runs.items()}
    return runs


def load_dataset(cfg):
    """Load dataset from cfg"""
    dataset_class = getattr(stereo, cfg['name'])
    dataset_kwargs = cfg.get('kwargs', {})
    dataset = dataset_class(**dataset_kwargs)
    return dataset


def write_graphs(idx_to_g: dict, file_root: str):
    """Write graphs into txt files"""
    for idx in range(len(idx_to_g)):
        with open(f'{file_root}-{idx}.txt', 'w') as f:
            f.write(str(idx_to_g[idx][0]))
            for v in idx_to_g[idx][1:]:
                f.write('\n')
                f.write(str(v))


def aggregate_As(
    As,
    tfs,
    genes,
    consensus: int = 1,
    top_tfs: int | None = None,
    top_genes_per_tf: int | None = None,
):
    """Compute the top edges under consensus

    As: array of shape (runs, T, targets, sources)
    """
    runs, T, n_targets, n_sources = As.shape
    assert n_targets == genes.size
    assert n_sources == tfs.size

    As_bin = (As != 0).sum(0)

    if top_tfs is not None:
        As = As.copy()
        As[:, As_bin < consensus] = 0
        for t in range(T):
            tf_sums = As[:, t].mean(0).sum(0)  # sum across runs and targets
            to_drop = tf_sums.argsort()[:-top_tfs]
            As[:, t, :, to_drop] = 0
        As_bin = (As != 0).sum(0)

    if top_genes_per_tf is not None:
        As = As.copy()
        As_mean = np.abs(As).mean(0)  # abs for tvdbn
        for t in range(T):
            A = As_mean[t]
            to_drop = A.argsort(0)[:-top_genes_per_tf]
            # to_drop = A <= np.quantile(A, 0.98, axis=0)
            As[:, t, to_drop, np.arange(As.shape[3])] = 0
        As_bin = (As != 0).sum(0)

    As_mean = As.mean(0)

    t_to_selected = {}

    for t in range(T):
        Aidx = As_bin[t] >= consensus
        trg, src = Aidx.nonzero()
        weight = As_mean[t][trg, src]
        t_to_selected[t] = list(zip(tfs[src], genes[trg], weight))
        print(
            f"t={t}\t Selected {len(t_to_selected[t])} edges\t"
            f"N. TFs = {len(np.unique(src))}\t"
            f"N. genes = {len(np.unique(trg))}\t"
        )

    return t_to_selected
