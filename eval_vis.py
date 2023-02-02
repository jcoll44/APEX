import os
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from model.OCWM import OCWM
from utils import spatial_transform
from dataloader.P2D import P2DDataset
from dataloader.Sketchy import SketchyDataset
from torch.utils.data import DataLoader
from dataloader.Box2D import Box2D

parser = argparse.ArgumentParser(description='APEX')
parser.add_argument('--ex_id', default=63,
                    help='which experiment to eval.')
parser.add_argument('--ck', default=2070,type=int,
                    help='which checkpoint to eval.')
parser.add_argument('--data', default='Box2D',type=str,
                    help='which datasets')
#torch.manual_seed(1)
#np.random.seed(1)
# Make CUDA operations deterministic
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
args = parser.parse_args()
bs = 4
data_name = args.data
if args.data == 'P2D':
    dataset_train=P2DDataset('./data/P2D/data/train/')
    dataloader_train=DataLoader(dataset_train, batch_size=bs, shuffle=False,pin_memory=False)
    dataset_test=P2DDataset('./data/P2D/data/test/')
    dataloader_test=DataLoader(dataset_test, batch_size=bs, shuffle=False,pin_memory=False)
    L = 20
elif args.data == 'sk':
    dataset_train=SketchyDataset('./data/Sketchy/data/train/')
    dataloader_train=DataLoader(dataset_train, batch_size=bs, shuffle=False,pin_memory=False)
    dataset_test=SketchyDataset('./data/Sketchy/data/test/')
    dataloader_test=DataLoader(dataset_test, batch_size=bs, shuffle=False,pin_memory=False)
    L = 10
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
    dataloader_test = DataLoader(val_set, batch_size=bs, num_workers = workers)

eval_dir = './eval_pics/{}_{}'.format(args.data,args.ex_id,args.ck)
os.makedirs(eval_dir,exist_ok = True)
directory_ck = './checkpoints/{}/{}/'.format(args.data,args.ex_id)
resume_checkpoint = directory_ck+'net-{}.ckpt'.format(args.ck)
with open(directory_ck+'args.txt', 'r') as f:
    args.__dict__ = json.load(f)
args.data=data_name

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    model = OCWM(args)
    model = model.cuda()
else:
    model = OCWM(args)
model.eval()

if torch.cuda.is_available():
    checkpoint = torch.load(resume_checkpoint)
else:
    checkpoint = torch.load(resume_checkpoint, map_location='cpu')

checkpoint2 = {}
for key in checkpoint['model_state_dict']:
    key2 = key[7:]
    print(key)
    checkpoint2[key2] = checkpoint['model_state_dict'][key]


model.load_state_dict(checkpoint2)
iter_idx = checkpoint['iter_idx']
print("Starting eval at iter = {}".format(iter_idx))

with torch.no_grad():
    for video in dataloader_train:
        break
    if torch.cuda.is_available():
        video = video.cuda()
    log_loss= model(video,iter_idx)
    print('-------------train----------------')
    print('recon loss: ',float(log_loss.recon_loss))
    print('prior loss: ',float(log_loss.prior_loss))
    print('mask_entropy_loss: ',float(log_loss.mask_entropy_loss))

    logger = model.logger
    video_np = np.transpose(video.cpu().data.numpy(),[0,1,3,4,2])
    for i in range(args.L):
        obj_dict = logger[i]
        obj_mask_ds=obj_dict.mask[:,-16:]#bs,16,1,128,128
        obj_recon_ds=obj_dict.recon[:,-16:]
        obj_masked_recon_ds=obj_dict.masked_recon[:,-16:]
        z_where = obj_dict.z_where
        obj_num = z_where.shape[1]
        z_where = z_where.view(-1,4)
        glimpse = spatial_transform(torch.stack(obj_num*(video[:,i],), dim=1).reshape(-1, 3,128,128),
                                    z_where,
                                    (obj_num*4, 3, 64, 64),
                                    inverse=False)
        input_all = spatial_transform(glimpse,
                                     z_where,
                                     (obj_num*4, 3, 128, 128),
                                     inverse=True).view(4,-1,3,128,128)
        input_ds = input_all[:,-16:]
        prop_num_max = 0
        if i != 0:
            obj_mask_tr=obj_dict.mask[:,:-16]#[obj_num,1,128,128]xbs
            obj_recon_tr=obj_dict.recon[:,:-16]
            obj_masked_recon_tr=obj_dict.masked_recon[:,:-16]
            z_where_tr = obj_dict.z_where[:,:-16]
            input_tr = input_all[:,:-16]
            prop_num_max = obj_recon_tr.shape[1]
        mask_bg = obj_dict.mask_bg#bs,1,128,128
        recon_bg = obj_dict.recon_bg
        masked_recon_bg = obj_dict.masked_recon_bg
        for j in range(4):
            demo_img = np.zeros([128*4,128*(19+prop_num_max)+32,3])
            demo_img[:,:128] = np.concatenate([video_np[j][i]]*4,axis=0)
            if i != 0:
                recon_img = torch.sum(obj_masked_recon_ds[j],dim=0)+torch.sum(obj_masked_recon_tr[j],dim=0)+masked_recon_bg[j]
            else:
                recon_img = torch.sum(obj_masked_recon_ds[j],dim=0)+masked_recon_bg[j]
            recon_img_np = np.transpose(recon_img.cpu().data.numpy(),[1,2,0])
            demo_img[:,128:256] = np.concatenate([recon_img_np]*4,axis=0)
            if i != 0:
                prop_obj_num = obj_mask_tr[j].shape[0]
                for k in range(prop_obj_num):
                    mr = obj_masked_recon_tr[j][k]
                    mr_np = np.transpose(mr.cpu().data.numpy(),[1,2,0])
                    r = obj_recon_tr[j][k]
                    r_np = np.transpose(r.cpu().data.numpy(),[1,2,0])
                    m = obj_mask_tr[j][k]
                    m_np = np.transpose(m.cpu().data.numpy(),[1,2,0])
                    ip = input_tr[j][k]
                    input_np = np.transpose(ip.cpu().data.numpy(),[1,2,0])
                    demo_img[:128,128*(2+k):128*(3+k)] = mr_np
                    demo_img[128:256,128*(2+k):128*(3+k)] = r_np
                    demo_img[128*2:128*3,128*(2+k):128*(3+k)] = m_np
                    demo_img[128*3:128*4,128*(2+k):128*(3+k)] = input_np
            demo_img[:,-128*17-32:-128*17] = 1.
            
            for k in range(16):
                mr = obj_masked_recon_ds[j][k]
                mr_np = np.transpose(mr.cpu().data.numpy(),[1,2,0])
                r = obj_recon_ds[j][k]
                r_np = np.transpose(r.cpu().data.numpy(),[1,2,0])
                m = obj_mask_ds[j][k]
                m_np = np.transpose(m.cpu().data.numpy(),[1,2,0])
                ip = input_ds[j][k]
                input_np = np.transpose(ip.cpu().data.numpy(),[1,2,0])
                demo_img[:128,128*(-17+k):128*(-16+k)] = mr_np
                demo_img[128:256,128*(-17+k):128*(-16+k)] = r_np
                demo_img[128*2:128*3,128*(-17+k):128*(-16+k)] = m_np
                demo_img[128*3:128*4,128*(-17+k):128*(-16+k)] = input_np

            mr_bg = masked_recon_bg[j]
            mr_bg_np = np.transpose(mr_bg.cpu().data.numpy(),[1,2,0])
            r_bg = recon_bg[j]
            r_bg_np = np.transpose(r_bg.cpu().data.numpy(),[1,2,0])
            m_bg = mask_bg[j]
            m_bg_np = np.transpose(m_bg.cpu().data.numpy(),[1,2,0])
            demo_img[:128,-128:] = mr_bg_np
            demo_img[128:256,-128:] = r_bg_np
            demo_img[128*2:128*3,-128:] = m_bg_np
            demo_img = np.clip(demo_img,0.,1.)
            plt.imsave(eval_dir+'/train-{}-{}.jpg'.format(i,j),demo_img)


    for video in dataloader_test:
        break
    if torch.cuda.is_available():
        video = video.cuda()
    log_loss= model(video,iter_idx)
    print('-------------eval----------------')
    print('recon loss: ',float(log_loss.recon_loss))
    print('prior loss: ',float(log_loss.prior_loss))
    print('mask_entropy_loss: ',float(log_loss.mask_entropy_loss))
    video_np = np.transpose(video.cpu().data.numpy(),[0,1,3,4,2])
    logger = model.logger
    for i in range(args.L):
        obj_dict = logger[i]
        obj_mask_ds=obj_dict.mask[:,-16:]#bs,16,1,128,128
        obj_recon_ds=obj_dict.recon[:,-16:]
        obj_masked_recon_ds=obj_dict.masked_recon[:,-16:]
        z_where = obj_dict.z_where
        obj_num = z_where.shape[1]
        z_where = z_where.view(-1,4)
        glimpse = spatial_transform(torch.stack(obj_num*(video[:,i],), dim=1).reshape(-1, 3,128,128),
                                    z_where,
                                    (obj_num*4, 3, 64, 64),
                                    inverse=False)
        input_all = spatial_transform(glimpse,
                                     z_where,
                                     (obj_num*4, 3, 128, 128),
                                     inverse=True).view(4,-1,3,128,128)
        input_ds = input_all[:,-16:]
        prop_num_max = 0
        if i != 0:
            obj_mask_tr=obj_dict.mask[:,:-16]#[obj_num,1,128,128]xbs
            obj_recon_tr=obj_dict.recon[:,:-16]
            obj_masked_recon_tr=obj_dict.masked_recon[:,:-16]
            z_where_tr = obj_dict.z_where[:,:-16]
            input_tr = input_all[:,:-16]
            prop_num_max = obj_recon_tr.shape[1]
        mask_bg = obj_dict.mask_bg#bs,1,128,128
        recon_bg = obj_dict.recon_bg
        masked_recon_bg = obj_dict.masked_recon_bg
        for j in range(4):
            demo_img = np.zeros([128*4,128*(19+prop_num_max)+32,3])
            demo_img[:,:128] = np.concatenate([video_np[j][i]]*4,axis=0)
            if i != 0:
                recon_img = torch.sum(obj_masked_recon_ds[j],dim=0)+torch.sum(obj_masked_recon_tr[j],dim=0)+masked_recon_bg[j]
            else:
                recon_img = torch.sum(obj_masked_recon_ds[j],dim=0)+masked_recon_bg[j]
            recon_img_np = np.transpose(recon_img.cpu().data.numpy(),[1,2,0])
            demo_img[:,128:256] = np.concatenate([recon_img_np]*4,axis=0)
            if i != 0:
                prop_obj_num = obj_mask_tr[j].shape[0]
                for k in range(prop_obj_num):
                    mr = obj_masked_recon_tr[j][k]
                    mr_np = np.transpose(mr.cpu().data.numpy(),[1,2,0])
                    r = obj_recon_tr[j][k]
                    r_np = np.transpose(r.cpu().data.numpy(),[1,2,0])
                    m = obj_mask_tr[j][k]
                    m_np = np.transpose(m.cpu().data.numpy(),[1,2,0])
                    ip = input_tr[j][k]
                    input_np = np.transpose(ip.cpu().data.numpy(),[1,2,0])
                    demo_img[:128,128*(2+k):128*(3+k)] = mr_np
                    demo_img[128:256,128*(2+k):128*(3+k)] = r_np
                    demo_img[128*2:128*3,128*(2+k):128*(3+k)] = m_np
                    demo_img[128*3:128*4,128*(2+k):128*(3+k)] = input_np
            demo_img[:,-128*17-32:-128*17] = 1.
            
            for k in range(16):
                mr = obj_masked_recon_ds[j][k]
                mr_np = np.transpose(mr.cpu().data.numpy(),[1,2,0])
                r = obj_recon_ds[j][k]
                r_np = np.transpose(r.cpu().data.numpy(),[1,2,0])
                m = obj_mask_ds[j][k]
                m_np = np.transpose(m.cpu().data.numpy(),[1,2,0])
                ip = input_ds[j][k]
                input_np = np.transpose(ip.cpu().data.numpy(),[1,2,0])
                demo_img[:128,128*(-17+k):128*(-16+k)] = mr_np
                demo_img[128:256,128*(-17+k):128*(-16+k)] = r_np
                demo_img[128*2:128*3,128*(-17+k):128*(-16+k)] = m_np
                demo_img[128*3:128*4,128*(-17+k):128*(-16+k)] = input_np

            mr_bg = masked_recon_bg[j]
            mr_bg_np = np.transpose(mr_bg.cpu().data.numpy(),[1,2,0])
            r_bg = recon_bg[j]
            r_bg_np = np.transpose(r_bg.cpu().data.numpy(),[1,2,0])
            m_bg = mask_bg[j]
            m_bg_np = np.transpose(m_bg.cpu().data.numpy(),[1,2,0])
            demo_img[:128,-128:] = mr_bg_np
            demo_img[128:256,-128:] = r_bg_np
            demo_img[128*2:128*3,-128:] = m_bg_np
            demo_img = np.clip(demo_img,0.,1.)
            plt.imsave(eval_dir+'/eval-{}-{}.jpg'.format(i,j),demo_img)
