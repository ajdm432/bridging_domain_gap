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

if __name__ == '__main__':
    cfg = Config()

    load_weights = True
    weight_file = 'pipnet_epoch59_train1k-augmentations_none.pth'

    meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface('meanface.txt', cfg.num_nb)
    utils = Utils(cfg, reverse_index1, reverse_index2, max_len)
        
    if load_weights:
        resnet18 = models.resnet18(pretrained=cfg.pretrained)
        net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
        state_dict = torch.load(weight_file)
        net.load_state_dict(state_dict)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    net = net.to(device)
    
    nme = []
    count = 0

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])
    time_all = 0


    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    while True:
        ret, frame = cap.read()
        if ret == True:
            inputs = Image.fromarray(frame[:,:,::-1].astype('uint8'), 'RGB')
            inputs = preprocess(inputs).unsqueeze(0)
            inputs = inputs.to(device)
            lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = utils.forward_pip(net, inputs, cfg.input_size, cfg.net_stride, cfg.num_nb)
            lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
            tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
            tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
            tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
            tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
            lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
            lms_pred = lms_pred.cpu().numpy()
            lms_pred_merge = lms_pred_merge.cpu().numpy()
            for i in range(cfg.num_lms):
                x_pred = lms_pred_merge[i*2]*frame_width
                y_pred = lms_pred_merge[i*2+1]*frame_height
                cv2.circle(frame, (int(x_pred), int(y_pred)), 1, (0, 0, 255), 2)
                
            count += 1
            #cv2.imwrite('drive/MyDrive/test-video-frames/'+str(count)+'.jpg', frame)
            cv2.imshow('webcam', frame)
            #cv2.imshow('1', frame)
        else:
            break

        # press escape to exit
        if (cv2.waitKey(30) == 27):
            break

        
    cap.release()
    cv2.destroyAllWindows()
