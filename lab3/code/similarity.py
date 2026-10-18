import numpy as np

def similarity(labels1, labels2):
    k1 = max(labels1) + 1
    k2 = max(labels2) + 1

    table = np.bincount(
        np.array(labels1) * k2 + np.array(labels2),
        minlength=k1 * k2
    ).reshape(k1, k2)

    both = np.sum(table * (table - 1) // 2)
    same1 = np.sum(table.sum(axis=1) * (table.sum(axis=1) - 1) // 2)
    same2 = np.sum(table.sum(axis=0) * (table.sum(axis=0) - 1) // 2)

    return both / (same1 + same2 - both)