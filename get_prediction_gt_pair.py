import torch
from torch import optim
from model.OCWM import OCWM
from dataloader.P2D import P2DDataset
from torch.utils.data import DataLoader
from eval_utils import average_segcover,ari_score2,get_gt_masks,binarize_masks,rle_encode

import os
import json
import copy
import argparse
import numpy as np
from attrdict import AttrDict
torch.set_default_tensor_type('torch.cuda.FloatTensor')


parser = argparse.ArgumentParser(description='APEX')
parser.add_argument('--ex_id', default=0,
                    help='which experiment to eval.')
parser.add_argument('--ck', default=40000,type=int,
                    help='which checkpoint to eval.')
parser.add_argument('--data', default='P2D',type=str,
                    help='which datasets')
args = parser.parse_args()
ex_id = args.ex_id
ck=args.ck
#load checkpoint
directory_ck = './checkpoints/{}/{}/'.format(args.data,ex_id)
checkpoint = torch.load(directory_ck+'net-{}.ckpt'.format(ck))
with open(directory_ck+'args.txt', 'r') as f:
  args = json.load(f)
args = AttrDict(args)

model = OCWM(args)
model.load_state_dict(checkpoint['model_state_dict'])
model.cuda()
model.eval()
#dataloader and ground-truth masks
bs=4
dataset=P2DDataset('./data/P2D/data/test/')
dataloader=DataLoader(dataset, batch_size=bs, shuffle=False,pin_memory=False)

count=0
output_list=[]
trj_idx_list=[]
for batch_idx, tensors in enumerate(dataloader):
    video = tensors[0].cuda()
    trj_idx = tensors[-1]
    with torch.no_grad():
        output_log= model(video,ck)
    logger = model.logger    
    for b in range(bs):
        video_list=[]
        output_list.append(video_list)
        trj_idx_list.append(trj_idx[b].cpu().data.numpy())
    for i in range(video.shape[1]):    
        obj_dict=model.logger[i]
        obj_masks=obj_dict.mask
        mask_bg = obj_dict.mask_bg
        obj_id = obj_dict.obj_id.cpu().data.numpy()
        z_pres = obj_dict.z_pres
        for b in range(bs):
            #fg&bg
            posi_idx=z_pres[b]>0.5
            posi_idx=posi_idx.cpu().data.numpy()
            bg_mask = mask_bg[b]#1,128,128
            fg_mask = obj_masks[b].view(-1,128,128)#n,1,128,128->n,128,128
            all_mask = torch.cat([bg_mask,fg_mask],dim=0)#n+1,128,128
            binarized_masks = binarize_masks(all_mask)
            binarized_masks = np.array(binarized_masks.cpu().data.numpy()).astype(np.uint8)
            binarized_masks=np.concatenate([binarized_masks[:1],binarized_masks[1:][posi_idx]],axis=0)
            ids=obj_id[b][posi_idx]
            ids=np.concatenate([np.array([0]),ids])
            frame=dict()
            frame['masks']=[]
            frame['ids']=[]
            for j in range(binarized_masks.shape[0]):
                frame['masks'].append(rle_encode(binarized_masks[j]))
                frame['ids'].append(ids[j])
            output_list[count+b].append(frame)
    count+=bs
    print('GPU usage in train: ', torch.cuda.max_memory_allocated()/1048576)
    print(batch_idx)
out_dir = './tracking_results/{}/{}/'.format(args.data,ex_id)
os.makedirs(out_dir,exist_ok = True)
np.save(out_dir+'pred.npy',output_list)
np.save(out_dir+'idx_map.npy',trj_idx_list)
    
        




