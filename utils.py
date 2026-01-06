import torch
from texttable import Texttable
from sklearn import metrics
import numpy as np


def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def get_evaluation_results(labels_true, labels_pred, labels_prob=None):
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    P = metrics.precision_score(labels_true, labels_pred, average='macro', zero_division=0)
    R = metrics.recall_score(labels_true, labels_pred, average='macro', zero_division=0)
    F1 = metrics.f1_score(labels_true, labels_pred, average='macro', zero_division=0)

    # AUC requires predicted probabilities or decision values
    if labels_prob is not None:
        AUC = metrics.roc_auc_score(labels_true, labels_prob, multi_class='ovr', average='macro')
    else:
        AUC = None

    # Specificity and Sensitivity require binary or one-vs-rest format
    if len(np.unique(labels_true)) == 2:  # Binary classification
        TN, FP, FN, TP = metrics.confusion_matrix(labels_true, labels_pred).ravel()
        if (TN + FP) > 0:
            SPE = TN / (TN + FP)  # Specificity
        else:
            # print(f"1——TN = {TN}, FP = {FP}, FN = {FN}")
            SPE = float('nan')
        SEN = TP / (TP + FN)  # Sensitivity (same as Recall)
    else:  # For multiclass we could calculate per-class and average
        # print(f"the number of classes = {len(np.unique(labels_true))}")

        # Compute specificity for each class
        num_classes = len(np.unique(labels_true))
        conf_matrix = metrics.confusion_matrix(labels_true, labels_pred)
        SPEs = []
        for i in range(num_classes):
            TN = np.sum(np.delete(np.delete(conf_matrix, i, 0), i, 1))
            FP = conf_matrix[i].sum() - conf_matrix[i, i]
            if (TN + FP) > 0:
                SPE = TN / (TN + FP)
                SPEs.append(SPE)
            else:
                # print(f"For class {i}: TN = {TN}, FP = {FP}")
                SPEs.append(float('nan'))

        # Average specificity across all classes
        SPE = np.nanmean(SPEs)

        SEN = R  # Recall already calculated above

    return ACC, P, R, F1, AUC, SPE, SEN


def loss_dependence(emb1, emb2, dim):
    R = torch.eye(dim).cuda() - (1 / dim) * torch.ones(dim, dim).cuda()
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC


def compute_renormalized_adj(adj, device):
    adj_ = torch.eye(adj.shape[0]).to(device) + adj
    rowsum = torch.tensor(adj_.sum(1)).to(device)
    degree_mat_inv_sqrt = torch.diag(torch.pow(rowsum, -0.5)).to(device)  # degree matrix
    adj_hat = (degree_mat_inv_sqrt).mm(adj_).mm(degree_mat_inv_sqrt)
    return adj_hat
