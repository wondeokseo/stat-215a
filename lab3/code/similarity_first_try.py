import numpy as np

def similarity(labels1, labels2):
    table = np.zeros((max(labels1) + 1, max(labels2) + 1), dtype=int)

    for a, b in zip(labels1, labels2):
        table[a, b] += 1

    both = np.sum(table * (table - 1) // 2)
    same1 = np.sum(table.sum(axis=1) * (table.sum(axis=1) - 1) // 2)
    same2 = np.sum(table.sum(axis=0) * (table.sum(axis=0) - 1) // 2)
    return both / (same1 + same2 - both)