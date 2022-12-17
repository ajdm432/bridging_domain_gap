import torch
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as T

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.optim as one
import logging
import wandb

from tqdm import tqdm
from scipy.integrate import simps
from config import Config

class Utils():
    def __init__(self, cfg, reverse_index1, reverse_index2, max_len):
        self.cfg = cfg
        self.reverse_index1 = reverse_index1
        self.reverse_index2 = reverse_index2
        self.max_len = max_len

    def get_label(self, label_file, task_type=None):
        label_path = label_file
        with open(label_path, 'r') as f:
            labels = f.readlines()
        labels = [x.strip().split() for x in labels]
        if len(labels[0])==1:
            return labels

        labels_new = []
        for label in labels:
            image_name = label[0]
            target = label[1:]
            target = np.array([float(x) for x in target])
            if task_type is None:
                labels_new.append([image_name, target])
            else:
                labels_new.append([image_name, task_type, target])
        return labels_new


    def compute_loss_pip(self, outputs_map, outputs_local_x, outputs_local_y, outputs_nb_x, outputs_nb_y, labels_map, labels_local_x, labels_local_y, labels_nb_x, labels_nb_y,  criterion_cls, criterion_reg, num_nb):

        tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_map.size()
        labels_map = labels_map.view(tmp_batch*tmp_channel, -1)
        labels_max_ids = torch.argmax(labels_map, 1)
        labels_max_ids = labels_max_ids.view(-1, 1)
        labels_max_ids_nb = labels_max_ids.repeat(1, num_nb).view(-1, 1)

        outputs_local_x = outputs_local_x.view(tmp_batch*tmp_channel, -1)
        outputs_local_x_select = torch.gather(outputs_local_x, 1, labels_max_ids)
        outputs_local_y = outputs_local_y.view(tmp_batch*tmp_channel, -1)
        outputs_local_y_select = torch.gather(outputs_local_y, 1, labels_max_ids)
        outputs_nb_x = outputs_nb_x.view(tmp_batch*num_nb*tmp_channel, -1)
        outputs_nb_x_select = torch.gather(outputs_nb_x, 1, labels_max_ids_nb)
        outputs_nb_y = outputs_nb_y.view(tmp_batch*num_nb*tmp_channel, -1)
        outputs_nb_y_select = torch.gather(outputs_nb_y, 1, labels_max_ids_nb)

        labels_local_x = labels_local_x.view(tmp_batch*tmp_channel, -1)
        labels_local_x_select = torch.gather(labels_local_x, 1, labels_max_ids)
        labels_local_y = labels_local_y.view(tmp_batch*tmp_channel, -1)
        labels_local_y_select = torch.gather(labels_local_y, 1, labels_max_ids)
        labels_nb_x = labels_nb_x.view(tmp_batch*num_nb*tmp_channel, -1)
        labels_nb_x_select = torch.gather(labels_nb_x, 1, labels_max_ids_nb)
        labels_nb_y = labels_nb_y.view(tmp_batch*num_nb*tmp_channel, -1)
        labels_nb_y_select = torch.gather(labels_nb_y, 1, labels_max_ids_nb)

        labels_map = labels_map.view(tmp_batch, tmp_channel, tmp_height, tmp_width)
        loss_map = criterion_cls(outputs_map, labels_map)
        loss_x = criterion_reg(outputs_local_x_select, labels_local_x_select)
        loss_y = criterion_reg(outputs_local_y_select, labels_local_y_select)
        loss_nb_x = criterion_reg(outputs_nb_x_select, labels_nb_x_select)
        loss_nb_y = criterion_reg(outputs_nb_y_select, labels_nb_y_select)
        return loss_map, loss_x, loss_y, loss_nb_x, loss_nb_y

    def train_model(self, det_head, net, train_loader, valid_loader, criterion_cls, criterion_reg, cls_loss_weight, reg_loss_weight, num_nb, optimizer, num_epochs, scheduler, save_dir, save_interval, device):
        nmes = []
        losses = []
        val_losses = []
        wandb.init(project="pipnet18_domain_gap")
        wandb.config = {
            "learning_rate": self.cfg.init_lr,
            "epochs": num_epochs,
            "batch_size": self.cfg.batch_size
            }
        wandb.watch(net, criterion_cls, log="all", log_freq=1) #TODO might want to change log_freq?

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            logging.info('-' * 10)
            epoch_loss = 0.0
    
            epoch_loss = self.fit(net, train_loader, criterion_cls, criterion_reg, cls_loss_weight, reg_loss_weight, num_nb, optimizer, epoch, num_epochs, device)
            validation_loss, nme = self.validate(net, valid_loader, criterion_cls, criterion_reg, cls_loss_weight, reg_loss_weight, num_nb, optimizer, epoch, num_epochs, device)

            losses.append(epoch_loss)
            val_losses.append(validation_loss)
            nmes.append(nme)

            wandb.log({"epoch": (epoch+1),
                "train_loss": (epoch_loss),
                "val_loss": (validation_loss),
                "NME": (nme)})
            wandb.define_metric("epoch")

            if epoch%(save_interval-1) == 0 and epoch > 0:
                filename = os.path.join(save_dir, 'epoch%d.pth' % epoch)
                torch.save(net.state_dict(), filename)
                print(filename, 'saved')
            scheduler.step()

        
        plt.figure(figsize=(10, 7))
        plt.plot(losses, color='orange', label='train loss')
        plt.plot(val_losses, color='red', label='validataion loss')
        plt.plot(nmes, color='blue', label='NME')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        return net

    def fit(self, net, train_loader, criterion_cls, criterion_reg, cls_loss_weight, reg_loss_weight, num_nb, optimizer, epoch, num_epochs, device):
        print('Training')
        training_loss = 0.0
        net.train()
        for i, data in enumerate(train_loader):
        
            inputs, labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y, target = data
            target = target.to(device)
            inputs = inputs.to(device)
            labels_map = labels_map.to(device)
            labels_x = labels_x.to(device)
            labels_y = labels_y.to(device)
            labels_nb_x = labels_nb_x.to(device)
            labels_nb_y = labels_nb_y.to(device)
            outputs_map, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = net(inputs)
            loss_map, loss_x, loss_y, loss_nb_x, loss_nb_y = self.compute_loss_pip(outputs_map, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y, labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y, criterion_cls, criterion_reg, num_nb)
            loss = cls_loss_weight*loss_map + reg_loss_weight*loss_x + reg_loss_weight*loss_y + reg_loss_weight*loss_nb_x + reg_loss_weight*loss_nb_y

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%10 == 0:
                print('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total training loss: {:.6f}> <map loss: {:.6f}> <x loss: {:.6f}> <y loss: {:.6f}> <nbx loss: {:.6f}> <nby loss: {:.6f}>'.format(
                    epoch, num_epochs-1, i, len(train_loader)-1, loss.item(), cls_loss_weight*loss_map.item(), reg_loss_weight*loss_x.item(), reg_loss_weight*loss_y.item(), reg_loss_weight*loss_nb_x.item(), reg_loss_weight*loss_nb_y.item()))
                logging.info('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total training loss: {:.6f}> <map loss: {:.6f}> <x loss: {:.6f}> <y loss: {:.6f}> <nbx loss: {:.6f}> <nby loss: {:.6f}>'.format(
                    epoch, num_epochs-1, i, len(train_loader)-1, loss.item(), cls_loss_weight*loss_map.item(), reg_loss_weight*loss_x.item(), reg_loss_weight*loss_y.item(), reg_loss_weight*loss_nb_x.item(), reg_loss_weight*loss_nb_y.item()))
        
            training_loss += loss.item()

        return training_loss/len(train_loader)


    def validate(self, net, valid_loader, criterion_cls, criterion_reg, cls_loss_weight, reg_loss_weight, num_nb, optimizer, epoch, num_epochs, device):
        print('Validating')
        validation_loss = 0.0
        net.eval()
        nme = 0.0
        for i, data in enumerate(valid_loader):
                inputs, labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y, target = data
                target = target.to(device)
                inputs = inputs.to(device)
                labels_map = labels_map.to(device)
                labels_x = labels_x.to(device)
                labels_y = labels_y.to(device)
                labels_nb_x = labels_nb_x.to(device)
                labels_nb_y = labels_nb_y.to(device)
                outputs_map, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = net(inputs)
                loss_map, loss_x, loss_y, loss_nb_x, loss_nb_y = self.compute_loss_pip(outputs_map, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y, labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y, criterion_cls, criterion_reg, num_nb)
                loss = cls_loss_weight*loss_map + reg_loss_weight*loss_x + reg_loss_weight*loss_y + reg_loss_weight*loss_nb_x + reg_loss_weight*loss_nb_y

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if i==0:
                # visualize:
                    lms = self.get_landmarks(net, torch.unsqueeze(inputs[0].clone(), 0))
                    nme = self.compute_nme_runtime(lms, target[0])
                    print('NME: ', nme)
                    #self.show_points_validation(inputs[0], target[0], lms)

                if i%10 == 0:
                    print('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total validation loss: {:.6f}> <map loss: {:.6f}> <x loss: {:.6f}> <y loss: {:.6f}> <nbx loss: {:.6f}> <nby loss: {:.6f}>'.format(
                        epoch, num_epochs-1, i, len(valid_loader)-1, loss.item(), cls_loss_weight*loss_map.item(), reg_loss_weight*loss_x.item(), reg_loss_weight*loss_y.item(), reg_loss_weight*loss_nb_x.item(), reg_loss_weight*loss_nb_y.item()))
                    logging.info('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total validation loss: {:.6f}> <map loss: {:.6f}> <x loss: {:.6f}> <y loss: {:.6f}> <nbx loss: {:.6f}> <nby loss: {:.6f}>'.format(
                        epoch, num_epochs-1, i, len(valid_loader)-1, loss.item(), cls_loss_weight*loss_map.item(), reg_loss_weight*loss_x.item(), reg_loss_weight*loss_y.item(), reg_loss_weight*loss_nb_x.item(), reg_loss_weight*loss_nb_y.item()))
            
                validation_loss += loss.item()

        epoch_validation_loss = validation_loss/len(valid_loader)
        return epoch_validation_loss, nme


    def forward_pip(self, net, inputs, input_size, net_stride, num_nb):
        net.eval()
        with torch.no_grad():
            outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = net(inputs)
            tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()
            #print(tmp_batch)
            assert tmp_batch == 1

            outputs_cls = outputs_cls.view(tmp_batch*tmp_channel, -1)
            max_ids = torch.argmax(outputs_cls, 1)
            max_cls = torch.max(outputs_cls, 1)[0]
            max_ids = max_ids.view(-1, 1)
            max_ids_nb = max_ids.repeat(1, num_nb).view(-1, 1)

            outputs_x = outputs_x.view(tmp_batch*tmp_channel, -1)
            outputs_x_select = torch.gather(outputs_x, 1, max_ids)
            outputs_x_select = outputs_x_select.squeeze(1)
            outputs_y = outputs_y.view(tmp_batch*tmp_channel, -1)
            outputs_y_select = torch.gather(outputs_y, 1, max_ids)
            outputs_y_select = outputs_y_select.squeeze(1)

            outputs_nb_x = outputs_nb_x.view(tmp_batch*num_nb*tmp_channel, -1)
            outputs_nb_x_select = torch.gather(outputs_nb_x, 1, max_ids_nb)
            outputs_nb_x_select = outputs_nb_x_select.squeeze(1).view(-1, num_nb)
            outputs_nb_y = outputs_nb_y.view(tmp_batch*num_nb*tmp_channel, -1)
            outputs_nb_y_select = torch.gather(outputs_nb_y, 1, max_ids_nb)
            outputs_nb_y_select = outputs_nb_y_select.squeeze(1).view(-1, num_nb)

            tmp_x = (max_ids%tmp_width).view(-1,1).float()+outputs_x_select.view(-1,1)
            tmp_y = (max_ids//tmp_width).view(-1,1).float()+outputs_y_select.view(-1,1)
            tmp_x /= 1.0 * input_size / net_stride
            tmp_y /= 1.0 * input_size / net_stride

            tmp_nb_x = (max_ids%tmp_width).view(-1,1).float()+outputs_nb_x_select
            tmp_nb_y = (max_ids//tmp_width).view(-1,1).float()+outputs_nb_y_select
            tmp_nb_x = tmp_nb_x.view(-1, num_nb)
            tmp_nb_y = tmp_nb_y.view(-1, num_nb)
            tmp_nb_x /= 1.0 * input_size / net_stride
            tmp_nb_y /= 1.0 * input_size / net_stride

        return tmp_x, tmp_y, tmp_nb_x, tmp_nb_y, outputs_cls, max_cls


    def get_landmarks(self, net, inputs):

        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = self.forward_pip(net, inputs, self.cfg.input_size, self.cfg.net_stride, self.cfg.num_nb)

        # merge neighbor predictions
        lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
        tmp_nb_x = lms_pred_nb_x[self.reverse_index1, self.reverse_index2].view(self.cfg.num_lms, self.max_len)
        tmp_nb_y = lms_pred_nb_y[self.reverse_index1, self.reverse_index2].view(self.cfg.num_lms, self.max_len)
        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()

        return lms_pred_merge


    def compute_nme(self, lms_pred, lms_gt, norm):
        lms_pred = lms_pred.reshape((-1, 2))
        lms_gt = lms_gt.reshape((-1, 2))
        #print(np.linalg.norm(lms_pred - lms_gt, axis=1));
        nme = np.mean(np.linalg.norm(lms_pred - lms_gt, axis=1)) / norm 
        return nme

    def compute_nme_runtime(self, lms_pred, target):
        target = target.cuda().detach().cpu().clone().numpy()
        target = np.reshape(target,(len(target)//2, 2))
        
        preds = lms_pred.cuda().detach().cpu().clone().numpy()
        preds = np.reshape(preds, (len(preds)//2, 2))

        norm = np.linalg.norm(target[36, ] - target[45, ])

        nme = np.mean(np.linalg.norm(preds - target, axis=1)) / norm 
        return nme

    def compute_fr_and_auc(self, nmes, thres=0.1, step=0.0001):
        num_data = len(nmes)
        xs = np.arange(0, thres + step, step)
        ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
        fr = 1.0 - ys[-1]
        auc = simps(ys, x=xs) / thres
        return fr, auc


    def show_points_validation(self, image, target, lms):
        target = target.cuda().detach().cpu().clone().numpy()
        image = image.cuda().detach().cpu().clone()
        img = np.array(image, dtype='float32')
        img = np.transpose(img,(1,2,0))

        plt.imshow(img)
        target = np.reshape(target,(len(target)//2, 2))
        plt.plot(target[:,0]*self.cfg.input_size,target[:,1]*self.cfg.input_size, 'bo', markersize=2)


        preds = lms.cuda().detach().cpu().clone().numpy()
        preds = np.reshape(preds, (len(preds)//2, 2))
        plt.plot(preds[:,0]*self.cfg.input_size,preds[:,1]*self.cfg.input_size, 'ro', markersize=2)


        plt.show()
        return


    def dataset_keypoints_plot(self, data):
        plt.figure(figsize=(10, 10))
        for i in range(9):
            image, labels_map, labels_x, labels_y, labels_nb_x, labels_nb_y, target = data[i]

            img = np.array(image, dtype='float32')
            img = np.transpose(img,(1,2,0))

            plt.imshow(img)
            target = np.reshape(target,(len(target)//2, 2))
            plt.plot(target[:,0]*self.cfg.input_size,target[:,1]*self.cfg.input_size, 'bo', markersize=2)
            plt.show()

