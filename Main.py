import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import model.SwinU
import Trainer
from Config import get_config

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset_path',
    type=str,
    default='/home/vincent/Downloads/st/Task01_BrainTumour/sliced_dataset/tv3',
    help='root dir for data')
parser.add_argument(
    '--data_list_path',
    type=str,
    default=
    '/home/vincent/Downloads/st/Task01_BrainTumour/sliced_dataset/tv3/lists',
    help='list dir')
parser.add_argument(
    '--output_path',
    type=str,
    default='/home/vincent/Downloads/st/Task01_BrainTumour/sliced_dataset/ooo',
    help='output dir')
parser.add_argument('--num_classes',
                    type=int,
                    default=4,
                    help='output channel of network')
parser.add_argument('--epochs',
                    type=int,
                    default=200,
                    help='maximum epoch number to train')
parser.add_argument('--batch_size',
                    type=int,
                    default=3,
                    help='batch_size per gpu')
parser.add_argument('--base_lr',
                    type=float,
                    default=0.01,
                    help='segmentation network learning rate')

args = parser.parse_args()

config = get_config()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = False

    random.seed(2022)
    np.random.seed(2022)
    torch.manual_seed(2022)
    torch.cuda.manual_seed(2022)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    net = model.SwinU.SwinUnet(config, num_classes=args.num_classes).cuda()
    pretrained_path = "/home/vincent/Documents/swinunetv2/pretrained_swin_model/swinv2_tiny_patch4_window16_256.pth"
    net.load_from(pretrained_path)

    Trainer.train(args, net, args.output_path)
