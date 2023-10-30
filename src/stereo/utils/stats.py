import numpy as np
from scipy.stats import hypergeom


def htest(universe, draws, successes, verbose=True):
    """Runs hypergeometric test.

    Parameters
    ----------
    universe: array
        All the elementgs in the universe.
    draws: array
        Array with the elements selected. Should be a subset of universe.
    successes: array
        Array with all the true elements. Should be a subset of universe.

    Returns
    -------
    pval: float
    """
    assert np.in1d(successes, universe).all()
    assert np.in1d(draws, universe).all()
    assert np.unique(universe).size == universe.size
    assert np.unique(draws).size == draws.size
    assert np.unique(successes).size == successes.size

    M = len(universe)
    n = len(successes)
    N = len(draws)
    drawn_successes = np.intersect1d(draws, successes)
    k = len(drawn_successes)
    pval = hypergeom.sf(M=M, n=n, N=N, k=k)
    if verbose:
        print(f"{M=}, {n=}, {N=}, {k=}, {pval=}")
    return pval
