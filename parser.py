import argparse


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', type=str, default="dataset",
                        help='data set path')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default="p4cnn",
                        help="p4cnn or z2cnn")

    args = parser.parse_args()
    return args
