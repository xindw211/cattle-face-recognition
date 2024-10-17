
import argparse

import os

import torch
from backbones import get_model
from dataset import get_dataloader
import numpy as np

from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

from eval import verification
import importlib
import os.path as osp
import os
def get_config(config_file):
    assert config_file.startswith('configs/'), 'config file setting must start with configs/'
    temp_config_name = osp.basename(config_file)
    temp_module_name = osp.splitext(temp_config_name)[0]
    config = importlib.import_module("configs.base")
    cfg = config.config
    config = importlib.import_module("configs.%s" % temp_module_name)
    job_cfg = config.config
    cfg.update(job_cfg)
    if cfg.output is None:
        cfg.output = osp.join('work_dirs', temp_module_name)
    return cfg,cfg.output
def main(args):
    cfg,name = get_config(args.config)

    # 创建识别模型
    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[0], bucket_cap_mb=16,
        find_unused_parameters=True)
    backbone.register_comm_hook(None, fp16_compress_hook)
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()
    model_path = name+"/model.pt"
    backbone.load_state_dict(torch.load(model_path), strict=False)
    backbone.eval()

    ver_list = []
    for name in cfg.val_targets:
        path = os.path.join(cfg.rec, name + ".bin")
        if os.path.exists(path):
            data_set = verification.load_bin(path, (112,112))
            ver_list.append(data_set)
    tpr, fpr, precision, recall, accuracy, roc_auc, best_distances, tar, far = verification.test(
        ver_list[0], backbone, 10, 10)
    max_index = np.argmax(accuracy)
    min_index = np.argmin(accuracy)
    filtered_accuracy = np.delete(accuracy, [max_index, min_index])
    acc = np.mean(filtered_accuracy)
    pre = np.mean(precision)
    call = np.mean(recall)
    print('Accuracy-Flip: %1.5f',acc)
    print('precision: %1.5f+-%1.5f',pre)
    print('recall: %1.5f',call)
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("--config", type=str, help="py config file",default='configs/pyramid-new_gam-1226-500')
    main(parser.parse_args())