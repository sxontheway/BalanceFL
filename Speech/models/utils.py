from bisect import bisect_right
from utils.misc import source_import
import torch

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_epochs=5,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        print(self.base_lrs)
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def init_models_importlib(config):
    """
    using library importlib and create_model function
    """
    networks = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for key, val in config['networks'].items():
        def_file = val['def_file']
        model_args = val['params']
        networks[key] = source_import(def_file).create_model(**model_args)
        # networks[key] = torch.nn.DataParallel(networks[key])
        networks[key].to(device)
    return networks


def init_models(config):
    """
    Directly init the network. Do not use the "create_model" function
    """
    from models.model import BBN_ResNet_Cifar, BasicBlock, DotProduct_Classifier, ResNet, Bottleneck
    device = config["device"]
    network = {}
    feat_dim = 256
    
    # for CUB dataset
    if config["dataset"]["name"] == "CUB":
        if "ResNet10s" == config["networks"]["feat_model"]["def_file"]: 
            l2_norm = config["networks"]["feat_model"]["params"]["l2_norm"]
            network["feat_model"] = ResNet(BasicBlock, [1,1,1,1], l2_norm=l2_norm, channels=[24,48,96,192]).to(device)  # 3/8 resnet10
            feat_dim = 192
        elif "ResNet10h" == config["networks"]["feat_model"]["def_file"]: 
            l2_norm = config["networks"]["feat_model"]["params"]["l2_norm"]
            network["feat_model"] = ResNet(BasicBlock, [1,1,1,1], l2_norm=l2_norm, channels=[32,64,128,256]).to(device) # half resnet10
            feat_dim = 256
        elif "ResNet10" == config["networks"]["feat_model"]["def_file"]: 
            l2_norm = config["networks"]["feat_model"]["params"]["l2_norm"]
            network["feat_model"] = ResNet(BasicBlock, [1,1,1,1], l2_norm=l2_norm).to(device)   # [64,128,256,512]
        elif "ResNet18" == config["networks"]["feat_model"]["def_file"]: 
            l2_norm = config["networks"]["feat_model"]["params"]["l2_norm"]
            network["feat_model"] = ResNet(BasicBlock, [2,2,2,2], l2_norm=l2_norm).to(device)
        elif "ResNet34" == config["networks"]["feat_model"]["def_file"]: 
            l2_norm = config["networks"]["feat_model"]["params"]["l2_norm"]
            network["feat_model"] = ResNet(BasicBlock, [3,4,6,3], l2_norm=l2_norm).to(device)
        else:
            raise NotImplementedError

    # for CIFAR and Speech
    feat_dim = config["networks"]["feat_model"]["feat_dim"]
    if config["dataset"]["name"] in ["CIFAR10", "CIFAR100", "Speech"]:
        if config["dataset"]["name"] == "Speech":
            input_channel = 1
        else:
            input_channel = 3
        if "ResNet32" in config["networks"]["feat_model"]["def_file"]: 
            l2_norm = config["networks"]["feat_model"]["params"]["l2_norm"]
            network["feat_model"] = BBN_ResNet_Cifar(BasicBlock, [5,5,5], l2_norm, feat_dim, input_channel).to(device)
        elif "ResNet20" in config["networks"]["feat_model"]["def_file"]: 
            l2_norm = config["networks"]["feat_model"]["params"]["l2_norm"]
            network["feat_model"] = BBN_ResNet_Cifar(BasicBlock, [3,3,3], l2_norm, feat_dim, input_channel).to(device)
        elif "ResNet8" in config["networks"]["feat_model"]["def_file"]: 
            l2_norm = config["networks"]["feat_model"]["params"]["l2_norm"]
            network["feat_model"] = BBN_ResNet_Cifar(BasicBlock, [1,1,1], l2_norm, feat_dim, input_channel).to(device)
        else:
            raise NotImplementedError

    if "DotProductClassifier" in config["networks"]["classifier"]["def_file"]: 
        params = config["networks"]["classifier"]["params"]
        num_classes = params["num_classes"]
        l2_norm, bias, scale = params["l2_norm"], params["bias"], params["scale"]
        network["classifier"] = DotProduct_Classifier(
            feat_dim=feat_dim, num_classes=num_classes, 
            l2_norm=l2_norm, bias=bias, scale=scale
            ).to(device)
    return network


def init_per_classifier(config, cls_number):
    """
    Init personalized models
    """
    from models.model import DotProduct_Classifier
    assert cls_number>=2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if "DotProductClassifier" in config["networks"]["classifier"]["def_file"]: 
        params = config["networks"]["classifier"]["params"]
        feat_dim = params["feat_dim"]
        classifier = DotProduct_Classifier(feat_dim=feat_dim, num_classes=cls_number).to(device)
    return classifier


def init_criterion(config):
    criterion_defs = config['criterions']
    def_file = criterion_defs['def_file']
    loss_args = criterion_defs['loss_params']
    loss_args.update({"device": config["device"]})

    if def_file.find("LwF") > 0:    
        from loss.LwFloss import create_loss
    elif def_file.find("KD") > 0:    
        from loss.KDLoss import create_loss
    elif def_file.find("BalancedCE") > 0:
        from loss.BalancedCE import create_loss
    else:
        raise RuntimeError
    
    criterion = create_loss(**loss_args)
    # criterion = source_import(def_file).create_loss(**loss_args)
    return criterion


def init_optimizers(networks, config):
    '''
    Seperate backbone optimizer and classifier optimizer
    ---
    Argu:
        - networks: a dictionary
        -config
    Return:
        - optimizer_dict
    '''
    networks_defs = config['networks']
    optimizer_choice = config["metainfo"]["optimizer"]
    optim_params_dict = {}

    optimizer_dict = {}

    for key, val in networks_defs.items():

        # obtain optim_params_dict
        optim_params = val['optim_params']
        optim_params_dict[key] = {
            'params': networks[key].parameters(), 
            'lr': optim_params['lr'], 
            'momentum': optim_params['momentum'], 
            'weight_decay': optim_params['weight_decay']}

        # optimizer
        if optimizer_choice == 'adam':
            # print('=====> Using Adam optimizer')
            optimizer = torch.optim.Adam([optim_params_dict[key],])
        else:
            # print('=====> Using SGD optimizer')
            optimizer = torch.optim.SGD([optim_params_dict[key],])
        optimizer_dict[key] = optimizer        

        # weight freezing
        if 'fix' in val and val['fix']:
            for param_name, param in networks[key].named_parameters():
                # Freeze all parameters except final fc layer
                if 'fc' not in param_name:
                    param.requires_grad = False
            print('=====> Freezing: {}'.format(key))
        if 'fix_set' in val:
            for fix_layer in val['fix_set']:
                for param_name, param in networks[key].named_parameters():
                    if fix_layer == param_name:
                        param.requires_grad = False
                        print('=====> Freezing: {}'.format(param_name))
                        continue

    return optimizer_dict

