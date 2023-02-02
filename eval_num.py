import json
import copy
import torch
import argparse
import numpy as np
from attrdict import AttrDict

from model.OCWM import OCWM
from dataloader.P2D import P2DDataset
from torch.utils.data import DataLoader
from eval_utils import average_segcover,ari_score2,get_gt_masks
torch.set_default_tensor_type('torch.cuda.FloatTensor')
#parse arguments
parser = argparse.ArgumentParser(description='APEX')
parser.add_argument('--ex_id', default=0,
                    help='which experiment to eval.')
parser.add_argument('--ck', default=40000,type=int,
                    help='which checkpoint to eval.')
parser.add_argument('--data', default='P2D',type=str,
                    help='which datasets')
args = parser.parse_args()
#load checkpoint
ck=args.ck
ex_id = args.ex_id
directory_ck = './checkpoints/{}/{}/'.format(args.data,ex_id)
checkpoint = torch.load(directory_ck+'net-{}.ckpt'.format(ck))
with open(directory_ck+'args.txt', 'r') as f:
  args = json.load(f)
args = AttrDict(args)
#create&load model
model = OCWM(args)
model.load_state_dict(checkpoint['model_state_dict'])
model.cuda()
model.eval()
#dataloader
bs=4
dataset=P2DDataset('./data/P2D/data/test/')
dataloader=DataLoader(dataset, batch_size=bs, shuffle=False,pin_memory=False)
annotation_dir = './data/P2D/annotation/test/'
#evaluation starts.
count=0.
ari_all=0.
ari_all_fg = 0.
sc_all=0.
sc_all_fg = 0.
for batch_idx, tensors in enumerate(dataloader):
    video = tensors[0].cuda()
    trj_idx = tensors[-1]
    masks_gt_all,masks_gt_fg_all,masks_gt_bg_all=get_gt_masks(annotation_dir,trj_idx,bs)
    with torch.no_grad():
        output_log= model(video,ck)
    logger = model.logger
    for i in range(20):    
        obj_dict=model.logger[i]
        obj_masks=obj_dict.mask
        mask_bg = obj_dict.mask_bg
        for b in range(bs):
            #fg&bg
            bg_mask = mask_bg[b]#1,128,128
            fg_mask = obj_masks[b].view(-1,128,128)#n,1,128,128->n,128,128
            all_mask = torch.cat([fg_mask,bg_mask],dim=0)#n+1,128,128
            mask_np=all_mask.permute(1,2,0).cpu().data.numpy()#128,128,n+1
            mask_gt=masks_gt_all[b][i]#128,128,m
            ari = ari_score2(mask_gt,mask_np)
            ari_all+=ari
            segA = torch.Tensor(np.argmax(mask_gt, axis=-1)).view(1,1,128,128).long()
            segB = torch.Tensor(np.argmax(mask_np, axis=-1)).view(1,1,128,128).long()
            sc = average_segcover(segA,segB)
            sc_all+=sc
            #fg only
            mask_gt_bg=masks_gt_bg_all[b][i].reshape(-1)
            fg_idx = np.where(mask_gt_bg == 0)[0]
            mask_gt_fg=masks_gt_fg_all[b][i]
            mask_gt_fg_flatten = mask_gt_fg.reshape(-1,mask_gt_fg.shape[2])[fg_idx]
            fg_mask_np = fg_mask.permute(1,2,0).cpu().data.numpy()
            fg_mask_np_flatten = fg_mask_np.reshape(-1,fg_mask_np.shape[2])[fg_idx]
            ari_fg = ari_score2(mask_gt_fg_flatten,fg_mask_np_flatten)
            ari_all_fg+=ari_fg
            bg_idx = np.where(masks_gt_bg_all[b][i]!=0)
            segA_fg = torch.Tensor(np.argmax(mask_gt, axis=-1)).view(1,1,128,128).long()
            segA_fg[:,:,bg_idx[0],bg_idx[1]]=-1
            segB_fg = torch.Tensor(np.argmax(mask_np, axis=-1)).view(1,1,128,128).long()
            sc_fg = average_segcover(segA_fg,segB_fg)
            sc_all_fg+=sc_fg
            count+=1
    print('GPU usage in train: ', torch.cuda.max_memory_allocated()/1048576)
    print(batch_idx)
ari_mean = ari_all/count
sc_mean =sc_all/count
ari_mean_fg = ari_all_fg/count
sc_mean_fg =sc_all_fg/count
print("ari_mean: ",ari_mean,"sc_mean: ",sc_mean)
print("ari_mean_fg: ",ari_mean_fg,"sc_mean_fg: ",sc_mean_fg)
np.save('./eval_results/{}_{}_{}.npy'.format(args.data,ex_id,ck),[[ari_mean,sc_mean],[ari_mean_fg,sc_mean_fg]])
    
        




