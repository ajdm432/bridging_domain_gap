import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.utils.data as data
import scipy
import torch.nn.functional as F
import torchvision.transforms as T

import pretrainedmodels
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import torch.optim as one
import matplotlib
import random
import time
import logging
import torchlm
import wandb

from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms
from scipy.integrate import simps
from scipy.stats import norm
from math import floor

from config import Config
from _data_utils import *
from _pipnet import *
from _utils import *

#---------------------------------------------------
if __name__ == '__main__':
    cfg = Config()
    cfg.experiment_name = "01"

    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    print(torch.cuda.is_available())

    #---------------------------------------------------

    # directory initiation:
    if not os.path.exists(os.path.join(cfg.log_dir, cfg.data_name)):
        os.mkdir(os.path.join(cfg.log_dir, cfg.data_name))

    save_dir = os.path.join(cfg.log_dir, cfg.data_name, cfg.experiment_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if not os.path.exists(os.path.join(save_dir, '/logs/')):
        os.mkdir(os.path.join(save_dir, '/logs/'))

    log_dir = os.path.join(save_dir, '/logs/', cfg.experiment_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # logging:
    logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO)

    #form_datafile(cfg)
    #generate_meanface()

    meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface('meanface.txt', cfg.num_nb)
    utils = Utils(cfg, reverse_index1, reverse_index2, max_len)

    if cfg.backbone == 'resnet18':
        resnet18 = models.resnet18(pretrained=cfg.pretrained)
        net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    criterion_cls = None
    if cfg.criterion_cls == 'l2':
        criterion_cls = nn.MSELoss()
    elif cfg.criterion_cls == 'l1':
        criterion_cls = nn.L1Loss()

    criterion_reg = None
    if cfg.criterion_reg == 'l1':
        criterion_reg = nn.L1Loss()
    elif cfg.criterion_reg == 'l2':
        criterion_reg = nn.MSELoss()


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    if cfg.pretrained:  
        optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr)
    else:
        optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.decay_steps, gamma=0.1)

    labels_train = utils.get_label("data_annotated.txt")
    labels_validate = utils.get_label("data_annotated_validation.txt")

    if cfg.det_head == 'pip':
        train_data = ImageFolder_pip(cfg.data_dir, 
                                labels_train, cfg.input_size, cfg.num_lms, 
                                cfg.net_stride, meanface_indices, True,
                                transforms.Compose([
                                transforms.RandomGrayscale(0.2),
                                transforms.Resize(cfg.input_size),
                                transforms.ToTensor(),
                                normalize]))
        
        valid_data = ImageFolder_pip(cfg.data_dir, 
                                labels_validate, cfg.input_size, cfg.num_lms, 
                                cfg.net_stride, meanface_indices, False,
                                transforms.Compose([
                                #transforms.RandomGrayscale(0.2),
                                transforms.Resize(cfg.input_size),
                                transforms.ToTensor(),
                                normalize
                                ]))
        
    else:
        print('No such head:', cfg.det_head)
        exit(0)


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)


    utils.train_model(cfg.det_head, net, train_loader, valid_loader, criterion_cls, criterion_reg, cfg.cls_loss_weight, cfg.reg_loss_weight, cfg.num_nb, optimizer, cfg.num_epochs, scheduler, save_dir, cfg.save_interval, device)

