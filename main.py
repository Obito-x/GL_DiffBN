import os
import warnings
import random
import torch
import configparser
import numpy as np
from args import parameter_parser
from utils import tab_printer
from train import train
from scipy.io import savemat
from plot_function import draw_tsne
import time

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    args = parameter_parser()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_ACC = []
    all_F1  = []
    all_AUC = []
    all_SEN = []
    all_SPE = []
    all_TIME = []

    # Repeat experiment n_repeated times
    for i in range(args.n_repeated):
        print(f"\n===== Run {i + 1}/{args.n_repeated} =====")
        ACC, F1, AUC, SEN, SPE, cost_time = train(args, device)
        all_ACC.append(ACC)
        all_F1.append(F1)
        all_AUC.append(AUC)
        if SEN is not None:
            all_SEN.append(SEN)
        if SPE is not None:
            all_SPE.append(SPE)
        all_TIME.append(cost_time)

    # Report mean and std over runs
    print("\n==================== Final Results ====================")
    print(f"ACC: {np.mean(all_ACC):.4f} ± {np.std(all_ACC):.4f}")
    print(f"F1:  {np.mean(all_F1):.4f}  ± {np.std(all_F1):.4f}")
    print(f"AUC: {np.mean(all_AUC):.4f} ± {np.std(all_AUC):.4f}")

    if all_SEN:
        print(f"SEN: {np.mean(all_SEN):.4f} ± {np.std(all_SEN):.4f}")
    if all_SPE:
        print(f"SPE: {np.mean(all_SPE):.4f} ± {np.std(all_SPE):.4f}")

    print(f"Time: {np.mean(all_TIME):.2f} ± {np.std(all_TIME):.2f} s")
    print("=======================================================")