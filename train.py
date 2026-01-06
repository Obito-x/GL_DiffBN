import warnings
import time
import random
from plot_function import permute_adj
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from args import parameter_parser
from utils import tab_printer, get_evaluation_results, compute_renormalized_adj
from Dataloader import load_data
from torch.autograd import Variable
from model import GL_DiffBN
import scipy.sparse as ss
from plot_function import draw_tsne
import scipy.sparse as sp
from sklearn.metrics import f1_score
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score


def train(args, device):
    feature_list, labels, idx_train, idx_val, idx_test = load_data(args, device)
    num_classes = len(np.unique(labels.cpu()))

    labels = labels.to(device)
    # feature_list = [feature_list[0]]
    num_view = len(feature_list)
    print(f"num_view: {num_view}")
    input_dims = [feature_list[i].shape[2] for i in range(num_view)]

    model = GL_DiffBN(input_dims, args.hidden_channels, num_classes, num_view, args).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0

    begin_time = time.time()

    with tqdm(total=args.num_epoch, desc="Training") as pbar:
        for epoch in range(args.num_epoch):
            model.train()
            mid_output = []

            for i in idx_train:
                new_feature_list = [feature_list[v][i] for v in range(num_view)]
                output = model(new_feature_list)
                output = output.mean(dim=0)
                mid_output.append(output)
            mid_output = torch.stack(mid_output)

            optimizer.zero_grad()
            loss = loss_function(mid_output, labels[idx_train])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_output = []
                for j in idx_val:
                    new_feature_list2 = [feature_list[v][j] for v in range(num_view)]
                    output = model(new_feature_list2)
                    output = output.mean(dim=0)
                    val_output.append(output)
                val_output = torch.stack(val_output)

                val_pred = torch.argmax(val_output, 1).cpu().detach().numpy()
                val_labels = labels.cpu().detach().numpy()[idx_val]
                val_acc = accuracy_score(val_labels, val_pred)

                if num_classes == 2:
                    val_probs = torch.softmax(val_output, dim=1).cpu().detach().numpy()
                    val_AUC = roc_auc_score(val_labels, val_probs[:, 1])
                    val_SEN = recall_score(val_labels, val_pred, pos_label=1)
                    val_SPE = recall_score(val_labels, val_pred, pos_label=0)

                    print(
                        f"\nEpoch [{epoch + 1}/{args.num_epoch}] - Loss: {loss.item():.6f} | Val_Acc: {val_acc:.4f} | "
                        f"Val_AUC: {val_AUC:.4f} | Val_SEN: {val_SEN:.4f} | Val_SPE: {val_SPE:.4f}"
                    )
                else:
                    val_F1 = f1_score(val_labels, val_pred, average='macro')

                    print(
                        f"\nEpoch [{epoch + 1}/{args.num_epoch}] - Loss: {loss.item():.6f} | Val_Acc: {val_acc:.4f} | "
                        f"Val_F1: {val_F1:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
                    )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                    best_epoch = epoch

                scheduler.step(val_acc)

            pbar.set_postfix({'Loss': '{:.6f}'.format(loss.item()), 'Val_Acc': '{:.4f}'.format(val_acc)})
            pbar.update(1)

    model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        test_output = []
        for j in idx_test:
            new_feature_list2 = [feature_list[v][j] for v in range(num_view)]
            output = model(new_feature_list2)
            output = output.mean(dim=0)
            test_output.append(output)
        test_output = torch.stack(test_output)

        test_pred = torch.argmax(test_output, 1).cpu().detach().numpy()
        test_labels = labels.cpu().detach().numpy()[idx_test]
        test_probs = torch.softmax(test_output, dim=1).cpu().detach().numpy()

        ACC, P, R, F1, AUC, SPE, SEN = get_evaluation_results(test_labels, test_pred, test_probs)

        F1_weighted = f1_score(test_labels, test_pred, average='weighted')

    if num_classes == 2:
        print(f"Test Results - ACC: {ACC:.4f}, F1: {F1:.4f}, AUC: {AUC:.4f}, SEN: {SEN:.4f}, SPE: {SPE:.4f}")
    else:
        print(f"Test Results - ACC: {ACC:.4f}, F1: {F1:.4f}, AUC: {AUC:.4f}, SEN: {SEN:.4f}, SPE: {SPE:.4f}")

    cost_time = time.time() - begin_time
    print(f"\nTraining completed, time elapsed: {cost_time // 60:.0f} min {cost_time % 60:.0f} sec")
    print(f"Best validation accuracy (epoch {best_epoch}): {best_val_acc:.4f}")

    return ACC, F1, AUC, SEN, SPE, cost_time



