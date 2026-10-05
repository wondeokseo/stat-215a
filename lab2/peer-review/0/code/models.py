"""Dimension reduction and clustering models for Lab 2."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def make_projection(name, n_components=2, perplexity=30, random_state=215):
    """Create an unfitted PCA or t-SNE estimator with the lab's settings.

    Args:
        name (str): Either "PCA" or "t-SNE".
        n_components (int): Number of output dimensions. Default 2; the valid
            range depends on the selected method and the fitted data.
        perplexity (float): t-SNE neighborhood parameter, ignored for PCA.
            Must be positive and smaller than the number of fitted samples.
        random_state (int): t-SNE random seed, ignored for PCA. Default 215.

    Returns:
        sklearn.decomposition.PCA or sklearn.manifold.TSNE: Unfitted
        estimator. PCA uses the covariance_eigh solver. t-SNE uses PCA
        initialization, automatic learning rate, and 1,000 iterations.

    Raises:
        ValueError: The projection name is unrecognized.

    Notes:
        The caller prepares the input data and calls fit or fit_transform.
        PCA centers its input during fitting but does not standardize it;
        any scaling or preliminary dimension reduction is done separately.
    """
    if name == "PCA":
        model = PCA(n_components=n_components, svd_solver="covariance_eigh")
    elif name == "t-SNE":
        model = TSNE(n_components=n_components, perplexity=perplexity,
                     init="pca", learning_rate="auto", max_iter=1000,
                     random_state=random_state, n_jobs=4, verbose=1)
    else:
        raise ValueError("Unknown projection name: " + name)
    return model


def make_clustering(name, n_clusters=2, random_state=215):
    """Create an unfitted K-means or diagonal-covariance Gaussian mixture.

    Args:
        name (str): Either "K-means" or "GMM".
        n_clusters (int): Number of K-means clusters or GMM components.
        random_state (int): Seed controlling initialization. Default 215.

    Returns:
        sklearn.cluster.KMeans or sklearn.mixture.GaussianMixture: Unfitted
        estimator with at most 300 iterations. K-means uses k-means++ and ten
        initializations. GMM uses three initializations, diagonal covariance,
        and covariance regularization of 1e-6.

    Raises:
        ValueError: The clustering name is unrecognized.

    Notes:
        The caller prepares features and fits the estimator. GMM allows each
        component a different variance along each input axis, with zero
        within-component covariances. Estimator labels are zero-based; the
        notebook adds one for display. This function does not select K.
    """
    if name == "K-means":
        model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10,
                       max_iter=300, random_state=random_state)
    elif name == "GMM":
        model = GaussianMixture(n_components=n_clusters, covariance_type="diag",
                                n_init=3, max_iter=300, reg_covar=1e-6,
                                random_state=random_state)
    else:
        raise ValueError("Unknown clustering name: " + name)
    return model


def fit_resampled_clustering(name, answers, scores, positions, n_clusters=5,
                             random_state=215, refit_pca=False):
    """Fit clustering on selected rows and predict the entire analysis cohort.

    Args:
        name (str): Clustering method accepted by make_clustering: "K-means"
            or "GMM".
        answers (pandas.DataFrame): Binary answer features for the full cohort,
            in the same row order as scores. Used when refit_pca is True.
        scores (pandas.DataFrame): Original PCA scores. Their column count sets
            the number of components when PCA is refitted; otherwise their
            values are used directly.
        positions (array-like of int): Zero-based row positions used for
            fitting. Repeated positions retain their multiplicity, allowing
            bootstrap sampling with replacement.
        n_clusters (int): Number of clusters or mixture components. Default 5.
        random_state (int): Clustering initialization seed. Default 215.
        refit_pca (bool): If True, fit centered PCA on the selected answer rows
            and transform the full cohort before fitting clustering. If False,
            keep the supplied scores fixed. Default False.

    Returns:
        tuple: Fitted clustering estimator, a one-dimensional NumPy array of
        one-based labels for every cohort row, and the fitted PCA estimator
        (or None when refit_pca is False).

    Notes:
        Sampling affects fitting only; each original respondent receives one
        predicted label. This function does not compute out-of-bag metrics or
        align the arbitrary cluster labels with a reference partition.
    """
    projection = None
    if refit_pca:
        projection = make_projection("PCA", n_components=scores.shape[1])
        # Fit PCA on the sample, then use that same projection for everyone.
        projection.fit(answers.iloc[positions])
        all_scores = projection.transform(answers)
    else:
        all_scores = scores.to_numpy()

    model = make_clustering(name, n_clusters=n_clusters, random_state=random_state)
    model.fit(all_scores[positions])
    labels = model.predict(all_scores) + 1
    return model, labels, projection


def align_cluster_labels(reference, labels, n_clusters):
    """Match a partition to reference groups by maximizing summed Jaccard overlap.

    Args:
        reference (array-like of int): One-based reference labels for the
            evaluation cohort. Every label from 1 through n_clusters must
            occur at least once.
        labels (array-like of int): One-based labels from another partition,
            with the same length and respondent order as reference. Labels
            must lie between 1 and n_clusters; groups may be absent.
        n_clusters (int): Number of groups in each partition.

    Returns:
        tuple: One-dimensional NumPy array of labels mapped to the reference
        numbering, and an array of matched Jaccard overlaps in reference
        cluster order. Jaccard is intersection size divided by union size,
        ranging from zero (no shared members) to one (identical membership).

    Notes:
        A one-to-one assignment maximizes the sum of per-group overlaps,
        giving each reference group equal weight. Matching renames labels;
        it does not change memberships. Splits, merges, or missing groups can
        force poor matches, and a match alone does not establish validity.
    """
    reference = np.asarray(reference)
    labels = np.asarray(labels)
    counts = np.zeros((n_clusters, n_clusters), dtype=int)
    np.add.at(counts, (reference - 1, labels - 1), 1)
    # Subtract shared members so they are not counted twice in the union.
    union = counts.sum(axis=1)[:, None] + counts.sum(axis=0)[None, :] - counts
    overlaps = counts / union
    rows, columns = linear_sum_assignment(overlaps, maximize=True)
    mapping = np.empty(n_clusters, dtype=int)
    mapping[columns] = rows + 1
    return mapping[labels - 1], overlaps[rows, columns]
