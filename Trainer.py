import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from Utils import DiceLoss
from torchvision import transforms
import data.Dataset


def train(args, model, out_path):

    logging.basicConfig(filename=out_path + "/log.txt",
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_epoch = args.epochs
    model.train()

    dataset_train = data.Dataset.Synapse_dataset(
        base_dir=args.dataset_path,
        list_dir=args.data_list_path,
        split="train",
        transform=transforms.Compose(
            [data.Dataset.RandomGenerator(output_size=[240, 240])]))

    logging.info("The length of train set is: {}".format(len(dataset_train)))

    def worker_init_fn(worker_id):
        random.seed(2022 + worker_id)

    train_loader = DataLoader(dataset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=False)
    max_iterations = max_epoch * len(train_loader)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(),
                          lr=base_lr,
                          momentum=0.9,
                          weight_decay=0.0001)

    writer = SummaryWriter(out_path + '/log')

    iter_num = 0

    logging.info("{} iterations per epoch.".format(len(train_loader)))
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(train_loader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch[
                'label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations)**0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' %
                         (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 10 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1),
                                       dim=1,
                                       keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50,
                                 iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 20

        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(out_path,
                                          'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(out_path,
                                          'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"