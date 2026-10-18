import numpy as np
from sklearn.cluster import KMeans
from similarity import similarity


def run_once(X, k, m, seed):
    rng = np.random.default_rng(seed)
    n = len(X)

    idx1 = rng.choice(n, int(n * m), replace=False)
    idx2 = rng.choice(n, int(n * m), replace=False)

    labels1 = KMeans(n_clusters=k, random_state=seed).fit_predict(X[idx1])
    labels2 = KMeans(n_clusters=k, random_state=seed).fit_predict(X[idx2])

    common, i1, i2 = np.intersect1d(idx1, idx2, return_indices=True)

    return similarity(labels1[i1], labels2[i2])

from joblib import Parallel, delayed

def run_parallel(X, m=0.2, B=100, n_jobs=2):
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_once)(X, k, m, 100 + i)
        for k in range(2, 11)
        for i in range(B)
    )

    return np.array(results).reshape(9, B).T


import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    import pyreadr

    df = next(iter(pyreadr.read_r("../data/lingBinary.Rdata").values()))
    X = df.iloc[:, 6:].to_numpy()

    S = run_parallel(X, m=0.8, B=100, n_jobs=4)

    Path("../results").mkdir(exist_ok=True)
    pd.DataFrame(S, columns=range(2, 11)).to_csv(
        "../results/S.csv", index=False
    )
