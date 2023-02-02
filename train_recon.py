import torch
import torch.nn as nn
import torch.optim as optim
from model.OCWM import OCWM
from dataloader.P2D import P2DDataset
from dataloader.Sketchy import SketchyDataset
from dataloader.Box2D import Box2D
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import wandb
from sklearn.model_selection import train_test_split


import os
import sys
import time
import json
import argparse
import numpy as np
from attrdict import AttrDict

parser = argparse.ArgumentParser(description='APEX')
parser.add_argument('--wt_ds', default=1.5e-4,type=float,
                    help='weight of discover z_what prior loss')
parser.add_argument('--wt_tr', default=1.5e-4,type=float,
                    help='weight of tracking z_what prior loss')
parser.add_argument('--we_ds', default=15.,type=float,
                    help='weight of discover z_where prior loss')
parser.add_argument('--we_tr', default=15.,type=float,
                    help='weight of tracking z_where prior loss')
parser.add_argument('--pr_ds', default=42.,type=float,
                    help='weight of discover z_pres prior loss')
parser.add_argument('--pr_tr', default=5000000.0,type=float,
                    help='weight of tracking z_pres prior loss')
parser.add_argument('--wt_bg', default=0.5,type=float,
                    help='weight of background z_what prior loss')
parser.add_argument('--ew', default=1.0,type=float,
                    help='weight of entropy loss')
parser.add_argument('--zwd', default=8, type=int,
                    help='z what dimension')
parser.add_argument('--bg_std', default=0.04, type=float,
                    help='std of background')
parser.add_argument('--std', default=0.1, type=float,
                    help='std of discover')
parser.add_argument('--T', default=0.6, type=float,
                    help='threshhold for propagation')
parser.add_argument('--tr_sh', default=0.1, type=float,
                    help='tracking shift limit')
parser.add_argument('--bs', default=6, type=int,
                    help='batch size')
parser.add_argument('--max_obj', default=3, type=int,
                    help='maximum number of objects.')
parser.add_argument('--ex_id', default=63, type=int,
                    help='experiment id')
parser.add_argument('--data', default='Box2D', type=str,
                    help='name of the datasets.')
parser.add_argument('--resume', action='store_true', default=False,
                    help='whether resume')
parser.add_argument('--resume_background', action='store_true', default=False,
                    help='whether resume with learnt background paramaters.')
parser.add_argument('--seed', default=20, type=int,
                    help='Fixed random seed.')

args = parser.parse_args()

# Fix seeds. Always first thing to be done after parsing the config!
# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
# # Make CUDA operations deterministic
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# Dataloader
# torch.multiprocessing.set_start_method('spawn')# good solution !!!!
# try:
#     torch.multiprocessing.set_start_method('spawn',force=True)
# except RuntimeError:
#     pass


bs = args.bs
if args.data == 'P2D':
    dataset_train=P2DDataset('./data/P2D/data/train/')
    dataloader_train=DataLoader(dataset_train, batch_size=bs, shuffle=True,pin_memory=False,generator=torch.Generator(device='cuda'))
    dataset_val=P2DDataset('./data/P2D/data/val/')
    dataloader_val=DataLoader(dataset_val, batch_size=bs, shuffle=False,pin_memory=False)
    args.L = 20
elif args.data == 'sk':
    dataset_train=SketchyDataset('./data/Sketchy/data/train/')
    dataloader_train=DataLoader(dataset_train, batch_size=bs, shuffle=True,pin_memory=False)
    dataset_val=SketchyDataset('./data/Sketchy/data/val/')
    dataloader_val=DataLoader(dataset_val, batch_size=bs, shuffle=False,pin_memory=False)
    args.L = 10
elif args.data == 'Box2D':
    args.L = 20
    dataset=Box2D('../../dataset/isaac_cubes3/',args.L)
    # train_size = int(dataset.__len__()*0.6)
    # test_size = dataset.__len__()-train_size
    # print("Training Size: ", train_size)
    # print("Testing Size: ", test_size)
    # print("Splitting Dataset")
    # train_set, val_set = torch.utils.data.random_split(dataset, [train_size,test_size])
    train_set, val_set=torch.utils.data.random_split(dataset, [1200, 800])
    workers = 4
    print("Creating Dataloaders")
    dataloader_train = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers = workers,generator=torch.Generator(device='cuda'))
    dataloader_val = DataLoader(val_set, batch_size=bs, num_workers = workers)


# Checkpoints    
torch.set_default_tensor_type('torch.cuda.FloatTensor')
directory_output = './checkpoints/{}/{}/'.format(args.data,args.ex_id)
os.makedirs(directory_output,exist_ok = True)
if args.resume:
    with open(directory_output+'args.txt', 'r') as f:
        args.__dict__ = json.load(f)
    args.resume = True
    args.resume_background = False
    all_iters = np.array([int(x[4:-5]) for x in os.listdir(directory_output) if x.endswith('ckpt')])
else:
    with open(directory_output+'args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

wandb.init(project="Box2D APEX",
        tags=[],
        dir="wandb/",
        config={
            "batch_size": bs,
            "Description": "Acceleration with multi step prediction, with lstm",
        })

# Create model 
model = OCWM(args)
model = model.cuda()
model_dict = model.state_dict()
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)
model = torch.nn.DataParallel(model)
start_iter = 0
model.train()


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

if args.resume_background:
    directory_read = './checkpoints/{}/{}/'.format(args.data,48)
    with open(directory_read+'args.txt', 'r') as f:
        args.__dict__ = json.load(f)
    all_iters = np.array([int(x[4:-5]) for x in os.listdir(directory_read) if x.endswith('ckpt')])
    
    resume_checkpoint = directory_read + 'net-{}.ckpt'.format(np.max(all_iters))
    print(resume_checkpoint)
    checkpoint = torch.load(resume_checkpoint)
    # model.load_state_dict(checkpoint['model_state_dict'])
    checkpoint2 = {}
    for key in checkpoint['model_state_dict']:
        if "background_net" in key or "bg" in key or "convlstm" in key:
            checkpoint2[key] = checkpoint['model_state_dict'][key]
            print(key)
        else:
            checkpoint2[key] = model.state_dict()[key]
    # model_dict.update(checkpoint2)
    model.load_state_dict(checkpoint2)

    print("Params to learn:")
    params_to_update = []
    for name,param in model.named_parameters():
        if "background_net" in name or "bg" in name or "convlstm" in name:
            param.requires_grad = False
            print("\t",name)
        else:
            param.requires_grad = True
            params_to_update.append(param)
            print(name)


    # optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
    optimiser = optim.Adam(params_to_update, 1e-4)

else:
    optimiser = optim.Adam(model.parameters(), 1e-4)


# Try to restore model and optimiser from checkpoint
if args.resume:
    resume_checkpoint = directory_output + 'net-{}.ckpt'.format(np.max(all_iters))
    checkpoint = torch.load(resume_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
    iter_idx = checkpoint['iter_idx'] + 1
    start_iter = iter_idx
    print("Starting eval at iter = {}".format(iter_idx))




def eval_model(model,dataloader_val,args,iter_idx,epoch):
    timer_eval = time.time()
    model.eval()
    loss_log_eval = AttrDict()
    loss_log_eval.recon_loss = 0.
    loss_log_eval.prior_loss = 0.
    loss_log_eval.mask_entropy_loss = 0.
    loss_log_eval.kl_where_ds = 0.
    loss_log_eval.kl_what_ds = 0.
    loss_log_eval.kl_pres_ds = 0.
    loss_log_eval.kl_where_tr = 0.
    loss_log_eval.kl_what_tr = 0.
    loss_log_eval.kl_pres_tr = 0.
    loss_log_eval.kl_what_bg = 0.
    loss_log_eval.loss = 0.
    num_iter=1.
    for video in dataloader_val:
        video = video.cuda()
        loss_log= model(video,iter_idx)
        for key in list(loss_log_eval.keys()):
            loss_log_eval[key] += loss_log[key].cpu().data.numpy()
        num_iter+=1.
    speed_eval = (time.time()-timer_eval)/num_iter
    wandb.log({"Train Loss": loss_log.loss,
                "Recon Loss" : loss_log.recon_loss,
                "prior_loss" : loss_log.prior_loss,
                "mask_entropy_loss" : loss_log.mask_entropy_loss,
                "kl_where_ds" : loss_log.kl_where_ds,
                "kl_what_ds" : loss_log.kl_what_ds,
                "kl_pres_ds" : loss_log.kl_pres_ds,
                "kl_where_tr" : loss_log.kl_where_tr,
                "kl_what_tr" : loss_log.kl_what_tr,
                "kl_pres_tr" : loss_log.kl_pres_tr,
                "kl_what_bg" : loss_log.kl_what_bg,
                "bg loss" : loss_log.bg_loss,
                "Iteration": epoch
                })
    print('iter: ',iter_idx,
          " ".join(str(key)+': '+str(value/num_iter) for key, value in loss_log_eval.items()),
          'speed: {:.3f}s/iter'.format(speed_eval))
    print('GPU usage in eval start: ', torch.cuda.max_memory_allocated()/1048576)
    model.train()

flag = 0
timer_epoch = time.time()
iter_idx = start_iter
for epoch in range(1000):
    for video in dataloader_train:

        if iter_idx == 5:
            timer = time.time()
        if iter_idx == 6:
            print('1 iter takes {:.3f}s.'.format(time.time()-timer))
        video = video.cuda()
        # Forward propagation
        optimiser.zero_grad()
        loss_log= model(video,iter_idx)
        # Backprop and optimise
        # if iter_idx<100:
        #     print("Loss Background")
        #     loss_log.bg_loss.backward()
        # else:
        loss_log.loss.backward()
        optimiser.step()
        # Heartbeat log
        # if (iter_idx % 300 == 0):
        # Print output and write to file
        speed = (time.time()-timer_epoch)/500.0
        timer_epoch = time.time()
        print('GPU usage in train: ', torch.cuda.max_memory_allocated()/1048576)
        print('Epoch: ',epoch,
                " ".join(str(key)+': '+str(value.cpu().data.numpy()) for key, value in loss_log.items()),
                'speed: {:.3f}s/iter'.format(speed))
        wandb.log({"Train Loss": loss_log.loss,
                    "Recon Loss" : loss_log.recon_loss,
                    "prior_loss" : loss_log.prior_loss,
                    "mask_entropy_loss" : loss_log.mask_entropy_loss,
                    "kl_where_ds" : loss_log.kl_where_ds,
                    "kl_what_ds" : loss_log.kl_what_ds,
                    "kl_pres_ds" : loss_log.kl_pres_ds,
                    "kl_where_tr" : loss_log.kl_where_tr,
                    "kl_what_tr" : loss_log.kl_what_tr,
                    "kl_pres_tr" : loss_log.kl_pres_tr,
                    "kl_what_bg" : loss_log.kl_what_bg,
                    "bg loss" : loss_log.bg_loss,
                    "Iteration": epoch
                    })
        with torch.no_grad():
            eval_model(model,dataloader_val,args,iter_idx,epoch)

        # Save checkpoints
        if iter_idx % 10 == 0:
            ckpt_file = directory_output + 'net-{}.ckpt'.format(iter_idx)
            print("Saving model training checkpoint to: {}".format(ckpt_file))
            model_state_dict = model.state_dict()
            ckpt_dict = {'iter_idx': iter_idx,
                            'model_state_dict': model_state_dict,
                            'optimiser_state_dict': optimiser.state_dict(),
                            'loss_log': loss_log}
            torch.save(ckpt_dict, ckpt_file)
            if flag == 0:
                print('Training starts, GPU usage in train start: ', torch.cuda.max_memory_allocated()/1048576)
                flag = 666
            #update iter_idx
        iter_idx+=1

