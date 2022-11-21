import argparse
import logging
import os
import random
import sys

import data.Dataset
import model.Swinv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from Config import get_config
from torch.utils.data import DataLoader
from tqdm import tqdm
from Utils import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path',
                    type=str,
                    default='/data/zhanwei/ppp/dataset/Test/',
                    help='root dir for validation volume data')

parser.add_argument('--list_dir',
                    type=str,
                    default='/data/zhanwei/ppp/dataset/lists',
                    help='list dir')
parser.add_argument('--output_dir',
                    default='/data/zhanwei/ppp/val_o_2',
                    type=str,
                    help='output dir')

args = parser.parse_args()

config = get_config()


def inference(args, model):
    logging.basicConfig(filename="logooo2.txt",
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    db_test = data.Dataset.MSDDataset(base_dir=args.dataset_path,
                                      split="test_vol",
                                      list_dir=args.list_dir)
    testloader = DataLoader(db_test,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch[
            "label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image,
                                      label,
                                      model,
                                      classes=4,
                                      test_save_path=args.output_dir,
                                      case=case_name,
                                      z_spacing=1)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' %
                     (i_batch, case_name, np.mean(
                         metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, 4):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' %
                     (i, metric_list[i - 1][0], metric_list[i - 1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info(
        'Testing performance in best val model: mean_dice : %f mean_hd95 : %f'
        % (performance, mean_hd95))
    return "Testing Finished!"


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    cudnn.benchmark = True
    cudnn.deterministic = False

    random.seed(2022)
    np.random.seed(2022)
    torch.manual_seed(2022)
    torch.cuda.manual_seed(2022)

    net = model.Swinv2.Swinv2Unet(config, num_classes=4).cuda()
    pretrained_path = "/home/zhanwei/datafolder/ppp/out_test2/epoch_299.pth"
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    print(net.load_state_dict(pretrained_dict["model"]))

    inference(args, net)
