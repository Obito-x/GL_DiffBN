# ADNI dataset processing
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import random
import time
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
import argparse


# def remove_common_features(features, n_remove=40):
#     """Discard common features from both high and low ends"""
#     print("Starting common feature discarding...")
#
#     n_samples, n_nodes, n_timepoints = features.shape
#     node_feature_matrix = []
#
#     # Reorganize features by node
#     for i in range(n_nodes):
#         columns_data = []
#         for sample_idx in range(n_samples):
#             column = features[sample_idx, i, :]
#             columns_data.append(column)
#         new_matrix = np.vstack(columns_data).T
#         node_feature_matrix.append(new_matrix)
#
#     # Perform SVD decomposition for each node
#     left_singular_matrices = []
#     for matrix in node_feature_matrix:
#         matrix_tensor = torch.Tensor(matrix)
#         u, s, v = torch.linalg.svd(matrix_tensor)
#         u[u < 0] = 0
#         left_singular_matrices.append(u)
#
#     # Extract feature weights
#     columns = []
#     for mat in left_singular_matrices:
#         column = mat[:, 0]
#         columns.append(column)
#     feature_weight = np.vstack(columns).T
#
#     # Calculate row means and sort
#     row_mean = np.mean(feature_weight, axis=1)
#     sorted_indices = np.argsort(row_mean)
#
#     # Select feature indices to discard
#     top_n_indices = sorted_indices[-n_remove:][::-1]  # Top n_remove indices
#     bottom_n_indices = sorted_indices[:n_remove]  # Bottom n_remove indices
#     delete_indices = np.concatenate((top_n_indices, bottom_n_indices))
#
#     print(f"Discarded feature indices: {delete_indices}")
#     print(f"Number of features discarded: {len(delete_indices)}")
#
#     # Remove selected features from original data
#     features_processed = np.delete(features, delete_indices, axis=2)
#     print(f"Original feature shape: {features.shape}, Processed feature shape: {features_processed.shape}")
#
#     return features_processed, delete_indices
#
#
# def retain_top_features(features, n_remove=40):
#     """Control experiment: retain top-λ features, discard only bottom-λ"""
#     print("Starting 【Retain top-λ】 control experiment...")
#
#     n_samples, n_nodes, n_timepoints = features.shape
#     node_feature_matrix = []
#     for i in range(n_nodes):
#         cols = [features[sample_idx, i, :] for sample_idx in range(n_samples)]
#         node_feature_matrix.append(np.vstack(cols).T)
#
#     left_singular_matrices = []
#     for mat in node_feature_matrix:
#         u, _, _ = torch.linalg.svd(torch.Tensor(mat))
#         u[u < 0] = 0
#         left_singular_matrices.append(u)
#
#     columns = [mat[:, 0] for mat in left_singular_matrices]
#     feature_weight = np.vstack(columns).T
#     row_mean = np.mean(feature_weight, axis=1)
#     sorted_indices = np.argsort(row_mean)
#
#     # Key modification: discard only bottom-λ, retain top-λ
#     delete_indices = sorted_indices[:n_remove]  # Only lowest commonality features
#
#     print(f"Discarded bottom-{n_remove} indices: {delete_indices}")
#
#     features_processed = np.delete(features, delete_indices, axis=2)
#     print(f"Original shape: {features.shape}, Processed shape: {features_processed.shape}")
#     return features_processed, delete_indices


def remove_common_features(features, n_remove=40):
    """Control experiment: retain bottom-λ features, discard only top-λ"""
    print("Starting 【Retain bottom-λ】 control experiment...")

    n_samples, n_nodes, n_timepoints = features.shape
    node_feature_matrix = []

    # Reorganize features by node (same as before)
    for i in range(n_nodes):
        columns_data = []
        for sample_idx in range(n_samples):
            column = features[sample_idx, i, :]
            columns_data.append(column)
        new_matrix = np.vstack(columns_data).T
        node_feature_matrix.append(new_matrix)

    # Perform SVD decomposition for each node
    left_singular_matrices = []
    for matrix in node_feature_matrix:
        matrix_tensor = torch.Tensor(matrix)
        u, s, v = torch.linalg.svd(matrix_tensor)
        u[u < 0] = 0
        left_singular_matrices.append(u)

    # Extract feature weights from first singular vector
    columns = []
    for mat in left_singular_matrices:
        column = mat[:, 0]
        columns.append(column)
    feature_weight = np.vstack(columns).T

    # Calculate row means and sort
    row_mean = np.mean(feature_weight, axis=1)
    sorted_indices = np.argsort(row_mean)

    # Key modification: discard only top-λ, retain bottom-λ
    # Select the indices with highest mean values (top n_remove)
    delete_indices = sorted_indices[-n_remove:][::-1]  # Top n_remove indices (descending order)

    print(f"Discarded top-{n_remove} indices: {delete_indices}")

    # Remove selected features from original data
    features_processed = np.delete(features, delete_indices, axis=2)
    print(f"Original shape: {features.shape}, Processed shape: {features_processed.shape}")

    return features_processed, delete_indices

def load_data(args, device):
    # Load raw time-series data (N, time, ROI)
    data1 = np.load('ADNI.npy', allow_pickle=True).item()
    final_timeseries = data1["timeseries"].squeeze(0)

    # Trim or zero-pad every sample to 90 time-points × 90 ROIs
    processed_arrays = []
    for arr in final_timeseries:
        if arr.ndim < 2:               # Skip corrupted slices
            continue
        arr = arr[:90, :90] if arr.shape[0] > 90 or arr.shape[1] > 90 else \
              np.pad(arr, [(0, max(0, 90-arr.shape[0])),
                          (0, max(0, 90-arr.shape[1]))], 'constant')
        processed_arrays.append(arr)
    data1['data'] = np.stack(processed_arrays)          # Shape: (N, 90, 90)

    # Build functional-connectivity matrices (Pearson correlation between ROIs)
    fc_data = np.array([np.corrcoef(sample.T) for sample in data1['data']])
    fc_data = np.nan_to_num(fc_data)
    labels = data1['label'].flatten() - data1['label'].min()

    # Train / val / test split
    idx_train, idx_val, idx_test = generate_partition(
        labels, args.train_ratio, args.val_ratio,
        1-args.train_ratio-args.val_ratio, args.shuffle_seed)
    labels = torch.from_numpy(labels).long().to(device)

    # Normalise time-series: z-score channel-wise using training stats
    train_ts = data1['data'][idx_train]                # (N_train, 90, 90)
    train_ts_t = train_ts.transpose(0, 2, 1)           # (N, 90, 90)  ROI×time
    flat = train_ts_t.reshape(-1, train_ts_t.shape[-1])
    mean_ts, std_ts = flat.mean(0), flat.std(0)
    std_ts[std_ts == 0] = 1
    norm_ts = ((data1['data'].transpose(0, 2, 1) - mean_ts) / std_ts).transpose(0, 2, 1)

    # Optional: remove the n_remove most common temporal features
    if getattr(args, 'use_common_feature_removal', False):
        n_remove = getattr(args, 'n_remove_features', 40)
        _, deleted_idx = remove_common_features(norm_ts[idx_train].transpose(0, 2, 1),
                                               n_remove)
        norm_ts = np.delete(norm_ts, deleted_idx, axis=2)  # Apply same mask to all sets

    # Normalise FC matrices using training mean & std
    train_fc = fc_data[idx_train]
    mean_fc, std_fc = train_fc.mean(), train_fc.std()
    std_fc = 1 if std_fc == 0 else std_fc
    norm_fc = (fc_data - mean_fc) / std_fc

    # Package features and return
    feature_list = [
        torch.stack([torch.from_numpy(norm_ts[i]).float() for i in range(len(labels))]).to(device),
        torch.stack([torch.from_numpy(norm_fc[i]).float() for i in range(len(labels))]).to(device)
    ]
    return feature_list, labels, idx_train, idx_val, idx_test

def generate_partition(labels, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=0):
    """
    Create stratified train / validation / test splits.
    """
    print(f"train_ratio = {train_ratio}, val_ratio = {val_ratio}, test_ratio = {test_ratio}")
    assert train_ratio + val_ratio + test_ratio == 1.0

    # Compute class counts
    each_class_num = count_each_class_num(labels)
    train_idx, val_idx, test_idx = [], [], []

    # Shuffle indices while keeping the random seed fixed
    random.seed(seed)
    indices = np.arange(len(labels))
    random.shuffle(indices)
    labels_shuffled = labels[indices]

    # For every class, split its samples proportionally
    for label in np.unique(labels):
        class_indices = indices[labels_shuffled == label]
        n_total = len(class_indices)

        n_train = int(train_ratio * n_total)
        n_val   = int(val_ratio * n_total)
        n_test  = n_total - n_train - n_val

        train_idx.extend(class_indices[:n_train])
        val_idx.extend(class_indices[n_train:n_train + n_val])
        test_idx.extend(class_indices[n_train + n_val:])

    return train_idx, val_idx, test_idx


def count_each_class_num(labels):
    """
    Count how many samples belong to each unique class.
    """
    count_dict = {}
    for label in labels:
        count_dict[label] = count_dict.get(label, 0) + 1
    return count_dict