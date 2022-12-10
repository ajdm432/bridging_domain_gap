import numpy as np
import os
import torch
import torch.utils.data as data
import cv2

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from config import Config
from _data_utils import *
from _pipnet import *
from _utils import *



#---------------------------------------------------

cfg = Config()
cfg.experiment_name = "01"

#---------------------------------------------------


RANDOM_SEED = 68

class FaceKeypointTestDataset(Dataset):
    def __init__(self, path):
      ind_names = np.array([])
      out_names = np.array([])
      for f in os.listdir(path + "/01_Indoor"):
        if not '.pts' in f:
          ind_names = np.append(ind_names, "/01_Indoor/" + f)
      for f in os.listdir(path + "/02_Outdoor"):
        if not '.pts' in f:
          out_names = np.append(out_names, '/02_Outdoor/' + f)
      self.ind_data = ind_names
      self.out_data = out_names
      self.path = path

    def __len__(self):
        return len(self.data)

    def get_300W_item(self, name):
      pathname = self.path + name
      image = cv2.imread(pathname)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      orig_h, orig_w, channel = image.shape
      image = cv2.resize(image, (cfg.input_size, cfg.input_size))

      idx = [idx for idx, x in enumerate(pathname) if x == '.']
      keyname = pathname[0:idx[0]] + ".pts"
      keypoints = np.loadtxt(keyname, dtype=float, comments=("version:", "n_points:", "{", "}"))
      # reshape the keypoints
      keypoints.reshape(68, 2)
      # normalize keypoints:
      keypoints = keypoints * [1 / orig_w, 1 / orig_h]
      return image, keypoints
      

    def get_300W_Data(self):
      np.random.seed(RANDOM_SEED)
      np.random.shuffle(self.ind_data)
      np.random.shuffle(self.out_data)
      #use a randomized 500 ims from outdoor images and a randomized 500 from indoor images
      data = np.append(self.ind_data[0:500], self.out_data[0:500]) 
      return data


save_dir = os.path.join(cfg.log_dir, cfg.data_name, cfg.experiment_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

load_weights = False


meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface('meanface.txt', cfg.num_nb)
utils = Utils(cfg, reverse_index1, reverse_index2, max_len)


if load_weights:
  resnet18 = models.resnet18(pretrained=cfg.pretrained)
  net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
  weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs-1))
  state_dict = torch.load(weight_file)
  net.load_state_dict(state_dict)

testdata = FaceKeypointTestDataset(cfg.test_data_dir)
dataframe = testdata.get_300W_Data()
nme = []
count = 0

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])

for n in dataframe:
  image, keypoints = testdata.get_300W_item(n)

  norm = np.linalg.norm(keypoints[36] - keypoints[45])

  inputs = Image.fromarray(image[:,:,::-1].astype('uint8'), 'RGB')
  inputs = preprocess(inputs).unsqueeze(0)
  inputs = inputs.to(device)

  #t1 = time.time()

  lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = utils.forward_pip(net, inputs, cfg.input_size, cfg.net_stride, cfg.num_nb)

  lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
  tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
  tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
  tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
  tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
  lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()

  #t2 = time.time()
  #time_all += (t2-t1)

  lms_pred = lms_pred.cpu().numpy()
  lms_pred_merge = lms_pred_merge.cpu().numpy()

  #nme_std = compute_nme(lms_pred, lms_gt, norm)
  #nmes_std.append(nme_std)
  lms_gt = np.reshape(keypoints, (1,keypoints.shape[0]*2))
  lms_gt = lms_gt[0]
  #print(lms_pred_merge.shape)
  #print(lms_gt.shape)
  nme_merge = utils.compute_nme(lms_pred_merge, lms_gt, norm)
  nme.append(nme_merge)

  preds = np.reshape(lms_pred, (len(lms_pred)//2, 2))
  truths = np.reshape(lms_gt, (len(lms_gt)//2, 2))

  if count % 20 == 0:
    plt.imshow(image)
    plt.plot(preds[:,0]*cfg.input_size,preds[:,1]*cfg.input_size, 'ro', markersize = 2)
    plt.plot(truths[:,0]*cfg.input_size,truths[:,1]*cfg.input_size, 'bo', markersize = 2)
    plt.show()  
    print("NME Value: " + str(nme_merge));

  count += 1


nme = np.array(nme)
print("Average NME: " + str(np.mean(nme)))