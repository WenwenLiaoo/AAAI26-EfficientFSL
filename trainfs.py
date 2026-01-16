import torch
from torch.optim import AdamW
from torch.nn import functional as F
from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
from timm.models import create_model
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from vtab import *
from utils import *
from efficientfsl import set_efficientfsl
from IPython import embed
import wandb
import torch.nn as nn
from loguru import logger
import sys
from torch.optim.lr_scheduler import StepLR
from peft import get_peft_model, LoraConfig, PrefixTuningConfig, AdaptionPromptConfig, PeftModel, TaskType

def train(config, model, dl, opt, scheduler, epoch, save_path):
    model.train()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    label = torch.arange(args.test_way).repeat(args.query).type(torch.cuda.LongTensor).to('cuda')

    best_model_wts = None
    best_acc = 0.0 
    
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        logger.info(f"Loaded model from {save_path}")

    for ep in range(epoch):
        pbar = tqdm(dynamic_ncols=True, leave=True, desc=False)
        model.train()
        model = model.cuda()
        total_loss = []
        total_acc = []
        total_size = len(dl)
        current_lr = opt.param_groups[0]['lr']
        print(f"Epoch {ep} Learning rate: {current_lr:.6f}")

        for i, (data, labels) in enumerate(tqdm(dl)):
            opt.zero_grad()
            pbar.set_description(f"Epoch {ep}/{epoch} | Training | Total Iter {total_size}")
            pbar.update(1)
            data = data.to('cuda')
            
            _,support,query = model(data)

            proto = support.view(args.shot, args.test_way, -1).mean(0)

            logit0, predict0 = Cosine_classifier(proto, query)
            loss = criterion(logit0, label)
            total_loss.append(loss.item())

            acc = (predict0 == label).float().mean()
            total_acc.append(acc.item())

            loss.backward()
            opt.step()

        logger.info(f"\nEpoch {ep} Loss: {sum(total_loss)/len(total_loss)} Acc: {sum(total_acc)/len(total_acc)*100:.2f}")

        if scheduler is not None:
            scheduler.step()

        acc, ci95 = test(model, val_dl)
        torch.cuda.empty_cache()
        logger.info(f"\nEpoch {ep} Acc: {acc:.2f} ± {ci95:.2f}")

        if acc > best_acc:
            best_acc = acc
            best_model_wts = model.state_dict().copy() 
            config['best_acc'] = acc
            logger.info("New best model found and saved!")

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), save_path)
        logger.info(f"Loaded best model with Val Acc: {best_acc}")

    acc, ci95 = test(model, test_dl)
    logger.info(f"Test Acc: {acc:.2f} ± {ci95:.2f}")

    model = model.cpu()
    return model

@torch.no_grad()
def test(model, dl):
    label = torch.arange(args.test_way).repeat(args.query).type(torch.cuda.LongTensor).to('cuda')
    total_test_acc = []
    model.eval()
    total_size = len(dl)
    pbar = tqdm(dynamic_ncols=True, leave=True, desc=False)
    model = model.cuda()
    
    for data, labels in tqdm(dl):
        pbar.set_description(f"Testing | Total Iter {total_size}")
        pbar.update(1)
        data = data.to('cuda')
        _, support, query = model(data)

        proto = support.view(args.shot, args.test_way, -1).mean(0)
        logit0, predict0 = Cosine_classifier(proto, query)
        acc = (predict0 == label).float().mean().item()
        total_test_acc.append(acc)

    mean_acc = np.mean(total_test_acc)*100
    ci95 = 1.96 * np.std(np.array(total_test_acc)) / np.sqrt(len(total_test_acc))*100  # 95% 置信区间
    
    logger.info(f"Test Acc: {mean_acc:.2f} ± {ci95:.2f}")
    
    return mean_acc, ci95

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-9)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--wd', type=float, default=1e-2)
    parser.add_argument('--model', type=str,
                        default='vit_base_patch16_224_in21k')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument("--method", type=str)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--feat_scale", type=float, default=1)
    parser.add_argument("--attn_scale", type=float, default=1)
    parser.add_argument("--ffn_scale", type=float, default=1)
    parser.add_argument("--token_nums", type=int, default=1)
    parser.add_argument("--search", type=int)
    parser.add_argument('--test-batch', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--save-path', type=str, default=None,
                        help='path to save best few-shot model checkpoint')
    args = parser.parse_args()
    fmt = "[{time: MM-DD hh:mm:ss}] {message}"
    logger_config = {
        "handlers": [
            {"sink": sys.stderr, "format": fmt},
        ],
    }
    logger.configure(**logger_config)
    config = get_config(args.ckpt, args.dataset)

    args.feat_scale = config["feat_scale"]
    args.attn_scale = config["attn_scale"]
    args.ffn_scale = config["ffn_scale"]
    args.token_nums = config["token_nums"]
    file_root = f"log/{args.dataset}/{args.exp_name}/feat_{args.feat_scale}_attn_{args.attn_scale}_ffn_{args.ffn_scale}_token_{args.token_nums}_lr_{args.lr}_wd_{args.wd}_{args.dataset}"
    file_path = f"{file_root}/{args.exp_name}.txt"
    if os.path.exists(file_path):
        print("file already exists")
        exit(0)
    if not os.path.exists(file_root):
        os.makedirs(file_root)
        open(file_path,'w').close()
    logger.add(file_path,format=fmt)    
    logger.info(args)
    set_seed(args.seed)
    if args.ckpt == 'vit_1k':
        checkpoint_path = "ckpt/1k.npz"
    else:
        checkpoint_path = "ckpt/21k.npz"

    # model = create_model("vit_small_patch16_224", pretrained=True)
    model = create_model(
        args.model, checkpoint_path=checkpoint_path, drop_path_rate=0.1)
    ref_model = create_model(
        args.model, checkpoint_path=checkpoint_path, drop_path_rate=0.1)
    
    train_dl, val_dl, test_dl = get_data(args.dataset, args.batch_size, args.num_workers,
                                 args.test_batch, args.test_way, args.shot, args.query)

    if args.method == 'efficientfsl':
        set_efficientfsl(args, model)
    else:
        exit()

    trainable = []

    config['best_acc'] = 0
    config['method'] = args.method

    if args.method == "efficientfsl":
        for n, p in model.named_parameters():
            if ('head' in n or 'efficientfsl' in n) and ('head.weight' not in n and 'head.bias' not in n):
                trainable.append(p)
            else:
                p.requires_grad = False
    else:
        exit()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable param: {n_parameters/1e6} M")
    opt = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scheduler = StepLR(opt, step_size=3, gamma=0.1) 

    if args.save_path is not None:
        save_path = args.save_path
    else:
        save_path = os.path.join(file_root, 'best_model.pth')

    model = train(config, model, train_dl,
                  opt, scheduler, epoch=args.num_epochs, save_path=save_path)
    logger.info(f"Best Acc: {config['best_acc']}")