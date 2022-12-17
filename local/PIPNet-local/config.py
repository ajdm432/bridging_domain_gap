

class Config():
    def __init__(self):
        self.det_head = 'pip'
        self.net_stride = 32
        self.batch_size = 16
        self.init_lr = 0.0001
        self.num_epochs = 60
        self.decay_steps = [30, 50]
        self.input_size = 256
        self.backbone = 'resnet18'
        self.pretrained = True
        self.criterion_cls = 'l2'
        self.criterion_reg = 'l1'
        self.cls_loss_weight = 10
        self.reg_loss_weight = 1
        self.num_lms = 68
        self.save_interval = 10
        self.num_nb = 10
        self.use_gpu = True
        self.gpu_id = 0
        self.data_dir = "C:/Users/trave/OneDrive/Documents/data/dataset_100000/"
        self.test_data_dir = "C:/Users/trave/OneDrive/Documents/data/300W"
        self.log_dir = "C:/Users/trave/OneDrive/Documents/data/saves/"
        self.data_name = "pipnet_rn18/"
        self.experiment_name = None