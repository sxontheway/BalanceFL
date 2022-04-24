import errno
import os, random
import importlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn


__all__ = ['mkdir_p', 'AverageMeter', 'update_config', 'source_import', 'check_nan']


def check_nan(weight, *args):
    num_nan = torch.sum(torch.isnan(weight))
    if num_nan!= 0:
        print(weight, num_nan, args)
        raise RuntimeError
        

def get_value(oldv, newv):
    if newv is not None:
        return newv
    else:
        return oldv


def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def update_config(config, args, log_dir):
    # Enable the args to overlay yaml configuration
    args_list = list(vars(args).keys())
    if "exp_name" in args_list:
        config['metainfo']['exp_name'] = get_value(config['metainfo']['exp_name'], args.exp_name)
    if "work_dir" in args_list:
        config['metainfo']['work_dir'] = get_value(config['metainfo']['work_dir'], args.work_dir)
    if "tao_ratio" in args_list:
        config['dataset']['tao_ratio'] = get_value(config['dataset']['tao_ratio'], args.tao_ratio)
    # config['dataset']['non_iidness'] = get_value(config['dataset']['non_iidness'], args.non_iidness)
    config['metainfo']['log_dir'] = log_dir
    return config



def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
def deterministic(seed):
    """
    Make the experiment reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
