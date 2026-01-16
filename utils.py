import torch
import random
import numpy as np
import yaml
import math

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def save(method, dataset, model, acc, ep):
    model.eval()
    model = model.cpu()
    trainable = {}
    for n, p in model.named_parameters():
        if 'adapter' in n or 'head' in n:
            trainable[n] = p.data
    torch.save(trainable, './models/%s/%s.pt'%(method, dataset))
    with open('./models/%s/%s.log'%(method, dataset), 'w') as f:
        f.write(str(ep)+' '+str(acc))
        

def load(method, dataset, model):
    model = model.cpu()
    st = torch.load('./models/%s/%s.pt'%(method, dataset))
    model.load_state_dict(st, False)
    return model

def get_config(method, dataset_name):
    with open('./configs/%s/%s.yaml'%(method, dataset_name), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def adjust_learning_rate(optimizer, epoch, total_epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (total_epoch - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

import torch.nn.functional as F
def Cosine_classifier(support, query, temperature=1):
    """Cosine classifier"""
    proto = F.normalize(support, dim=-1)
    query = F.normalize(query, dim=-1)
    logits = torch.mm(query, proto.permute([1, 0])) / temperature
    predict = torch.argmax(logits, dim=1)
    return logits, predict

from torchvision import transforms
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                         np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                         np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
])