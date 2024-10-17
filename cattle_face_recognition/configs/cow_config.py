# 2024.09.09
import os

from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 1.0, 0.0)
config.network = "new_gam_r100"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.batch_size = 128
config.dali = False

config.optimizer = "adam"
config.lr = 0.001
config.weight_decay = 0.001


config.rec = os.path.join(os.getcwd(), 'datasets/pyramid', 'train_cow')
config.num_classes = 522
config.num_image = 11134
config.num_epoch = 150
config.warmup_epoch = 25
config.val_targets = ['test_cow']

