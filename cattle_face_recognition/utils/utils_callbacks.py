import logging
import os
import time
from typing import List

import torch
import numpy as np
from eval import verification
from utils.utils_logging import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from torch import distributed
from utils.utils_config import get_config_name

class CallBackVerification(object):

    def __init__(self, val_targets, rec_prefix, summary_writer=None, image_size=(112, 112), wandb_logger=None):
        self.rank: int = distributed.get_rank()
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        if self.rank is 0:
            self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

        self.summary_writer = summary_writer
        self.wandb_logger = wandb_logger

    def ver_test(self, backbone: torch.nn.Module, epoch: int):
        results = []
        for i in range(len(self.ver_list)):

            accuracy = verification.test(self.ver_list[i], backbone, 10, 10)
            max_index = np.argmax(accuracy)  
            min_index = np.argmin(accuracy)
            filtered_accuracy = np.delete(accuracy, [max_index, min_index])
            acc = np.mean(filtered_accuracy)
           
            logging.info('[%s][%d]Accuracy-Flip: %1.5f' % (self.ver_name_list[i], epoch, acc))

            if acc > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc
            logging.info(
                '[%s][%d]New-Accuracy-Highest: %1.5f' % (self.ver_name_list[i], epoch, self.highest_acc_list[i]))

            name = get_config_name(self.config_name)

            if not os.path.exists('work_dirs/{}/logs'.format(name)):
                os.makedirs('work_dirs/{}/logs'.format(name))

            with open('work_dirs/{}/logs/log.txt'.format(name), 'a') as f:
                val_list = [
                    epoch,
                    acc,
                ]
                log = '\t'.join(str(value) for value in val_list)
                f.writelines(log + '\n')
            results.append(acc)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, epoch, backbone: torch.nn.Module,name):
        self.config_name = name
        if self.rank is 0 :
            backbone.eval()
            self.ver_test(backbone, epoch)
            backbone.train()

class CallBackLogging(object):
    def __init__(self, frequent, total_step, batch_size, start_step=0, writer=None):
        self.frequent: int = frequent
        self.rank: int = distributed.get_rank()
        self.world_size: int = distributed.get_world_size()
        self.time_start = time.time()
        self.total_step: int = total_step
        self.start_step: int = start_step
        self.batch_size: int = batch_size
        self.writer = writer
        self.init = False
        self.tic = 0

    def __call__(self,
                 step: int,
                 loss: AverageMeter,
                 epoch: int,
                 fp16: bool,
                 learning_rate: float,
                 grad_scaler: torch.cuda.amp.GradScaler,
                 ):
        if self.rank == 0 and step > 0 and step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                time_now = time.time()
                time_sec = int(time_now - self.time_start)
                time_sec_avg = time_sec / (step - self.start_step + 1)
                eta_sec = time_sec_avg * (self.total_step - step - 1)
                time_for_end = eta_sec / 3600

                if fp16:
                    msg = "Epoch: %d   Loss %.4f   LearningRate %.6f   Global Step: %d  Required: %1.f hours" % (epoch, loss.avg, learning_rate, step, time_for_end)

                else:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                          "Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, step, time_for_end
                          )
                logging.info(msg)
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()
