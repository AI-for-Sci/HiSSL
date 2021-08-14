import numpy as np

from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def cluster_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)

    # assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(y_pred[i]), int(y_true[i])] += 1

    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment
    ind = linear_sum_assignment(w.max() - w)
    ind_1 = np.concatenate((np.array([ind[0]]), np.array([ind[1]])), axis=0)
    ind_1 = np.transpose(ind_1)

    ac = sum([w[i, j] for i, j in ind_1]) * 1.0 / y_pred.size
    return ac


# # 聚类指标
ACC = cluster_accuracy

# Adjusted Rand index 调整兰德系数
ARI = adjusted_rand_score

# Mutual Information based scores 互信息
NMI = normalized_mutual_info_score

# Silhouette Coefficient 轮廓系数
SS = silhouette_score
