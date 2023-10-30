from typing import Dict, List

import torch
from scipy.stats import hypergeom
from torch import FloatTensor
from torchmetrics.functional import pearson_corrcoef, spearman_corrcoef
from torchmetrics.functional.classification import (
    binary_f1_score,
    binary_precision,
    binary_recall,
)


def A_seq_metrics(
    A_seq: List[FloatTensor],
    true_A_seq: List[FloatTensor],
    threshold: float = 0,
) -> Dict[str, float]:
    """General metrics between two adj matrices."""
    metrics = {}

    for i, (A, true_A) in enumerate(zip(A_seq, true_A_seq)):
        A = A.flatten()
        true_A = true_A.flatten()
        metrics[f"A.{i}.pearson"] = pearson_corrcoef(A, true_A)
        metrics[f"A.{i}.spearman"] = spearman_corrcoef(A, true_A)

        edge = A > threshold
        true_edge = true_A > threshold
        metrics[f"A.{i}.f1"] = binary_f1_score(edge, true_edge)
        metrics[f"A.{i}.recall"] = binary_recall(edge, true_edge)
        metrics[f"A.{i}.precision"] = binary_precision(edge, true_edge)

    return metrics


def A_seq_src_metrics(
    A_seq: List[FloatTensor],
    true_A_seq: List[FloatTensor],
    theta_seq: List[FloatTensor],
    mode: str = 'target_to_source',
) -> Dict[str, float]:
    """Metrics that evaluates the number of src nodes identified.
    """
    def metric_for_pred(pred, true, prefix=''):
        metrics = {}
        metrics[f"src.{i}.f1{prefix}"] = binary_f1_score(pred, true)
        metrics[f"src.{i}.recall{prefix}"] = binary_recall(pred, true)
        metrics[f"src.{i}.precision{prefix}"] = binary_precision(pred, true)
        overlap = torch.dot(true.long(), pred.long())
        sf = hypergeom.sf(
            M=len(true),  # total
            n=true.sum(),  # true total
            N=pred.sum(),  # draws
            k=overlap,  # successes
        )
        metrics[f"src.{i}.hypergeom{prefix}"] = sf
        metrics[f"src.{i}.overlap{prefix}"] = overlap.float()
        return metrics

    metrics = {}
    axis = 0 if mode == 'target_to_source' else 1
    true_sources = [A.sum(axis=axis) > 0 for A in true_A_seq]
    pred_sources = [A.sum(axis=axis) > 0 for A in A_seq]
    theta_sources = [theta > 0 for theta in theta_seq]
    zipped = zip(pred_sources, true_sources, theta_sources)
    for i, (pred_src, true_src, theta_src) in enumerate(zipped):
        metrics = {**metrics, **metric_for_pred(pred_src, true_src, '.A')}
        metrics = {**metrics, **metric_for_pred(theta_src, true_src, '.theta')}
    return metrics
