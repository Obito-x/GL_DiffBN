import argparse

from sympy import false


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="Device: cuda:num or cpu")
    parser.add_argument("--path", type=str, default='D:\Postgraduate\semster_2\summer\BaseLine\ECMGD_Code——ADNI\\', help="Path of datasets")

    parser.add_argument("--dataset", type=str, default="ADNI", help="Name of datasets")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--shuffle_seed", type=int, default=2, help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--fix_seed", action='store_true', default=True, help="Fix random seed")

    parser.add_argument('--hidden_channels', type=int, default=90, help="Number of hidden channels")
    parser.add_argument('--alpha', type=float, default=0.01, help='Weight for residual link')
    parser.add_argument('--layers', type=int, default=1, help='Number of layers')

    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')

    parser.add_argument("--n_repeated", type=int, default=2, help="Number of repeated times. Default is 10.")
    parser.add_argument("--save_results", action='store_true', default=True, help="Save results")
    parser.add_argument("--save_all", action='store_true', default=True, help="Save all results")
    parser.add_argument("--save_loss", action='store_true', default=True, help="Save loss")
    parser.add_argument("--save_ACC", action='store_true', default=True, help="Save accuracy")
    parser.add_argument("--save_F1", action='store_true', default=True, help="Save F1 score")

    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of train samples")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of validate samples")
    parser.add_argument("--num_epoch", type=int, default=1, help="Number of training epochs. Default is 200.")

    parser.add_argument("--use_common_feature_removal", action='store_true', default=True,
                       help="Whether to use common feature removal")
    parser.add_argument("--n_remove_features", type=int, default=5,
                       help="Number of features to remove from top and bottom (total removed = n_remove_features * 2)")

    args = parser.parse_args()

    return args