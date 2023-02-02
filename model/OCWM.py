import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from modules.Discover import Discover
from modules.tracking import Tracker
from modules.dla import ConvLSTMEncoder
from modules.background import Background
from modules.networks import GlimpseNet,MLP

import numpy as np
from attrdict import AttrDict
from utils import spatial_transform

class OCWM(nn.Module):

    def __init__(self,args):
        super(OCWM, self).__init__()
        # --- Inference ---
        self.convlstm = ConvLSTMEncoder(3)
        self.glimpse_net = GlimpseNet(args)

        self.discover_net = Discover(self.glimpse_net,args)
        self.tracker_net = Tracker(self.glimpse_net,args)
        self.background_net = Background(args)
        # --- constants ---
        self.args=args
        self.img_size=128
        self.bs = args.bs
        self.logger = []
        self.obj_count_flag = 2
        self.register_buffer('std',args.std * torch.ones(1, 1, 1, 1))
        self.register_buffer('bg_std',args.bg_std * torch.ones(1, 1, 1, 1))

    def forward(self, video, global_step):
        # --- collect returns ---
        self.bs = video.shape[0]
        recon_loss_all = 0
        prior_loss_all = 0
        mask_entropy_loss_all = 0
        logits_loss_all = 0.
        kl_what_ds_all=0.
        kl_what_tr_all=0.
        kl_where_ds_all=0.
        kl_where_tr_all=0.
        kl_pres_ds_all=0.
        kl_pres_tr_all=0.
        kl_what_bg_all=0.
        self.logger = []
        img_embd_video = self.convlstm(video)
        h_bg = None
        for i in range(self.args.L):
            img = video[:,i]
            if i == 0:
                log_s = torch.zeros([self.bs,1,self.img_size,self.img_size])
                # discover
                recon,mask_contri,mask_rectified,\
                z_where,z_what,z_pres,z_what_h,\
                kl_where_ds,kl_what_ds,kl_pres_ds,\
                glimpse_recon,glimpse_mask = self.discover_net(img,
                                                                         img_embd_video[:,i],
                                                                         log_s,
                                                                         global_step)
                # collect loss
                kl_where_ds_all += kl_where_ds
                kl_what_ds_all += kl_what_ds
                kl_pres_ds_all += kl_pres_ds
                # get normed masks
                mask_entropy_loss,mask,log_s1 = self.get_normed_masks(mask_contri,mask_rectified)
                obj_id=torch.arange(16)+1
                obj_id=obj_id.reshape(1,16).repeat(self.bs,1)
                log_dict = self.logging(recon,mask,z_where,z_what,z_pres,glimpse_recon,glimpse_mask,obj_id)
                self.logger.append(log_dict)
                # background
                h_bg,prior_bg,recon_bg = self.background_net(img,
                                                             log_s1,
                                                             h_bg,
                                                             self.logger)
                kl_what_bg_all+=prior_bg
                # loss
                bg_recon_loss = self.get_recon_loss_bg(img,recon_bg,log_s1.exp())
                recon_loss = self.get_recon_loss(img,recon,mask,recon_bg,log_s1.exp())
                prior_loss = kl_where_ds+kl_what_ds+kl_pres_ds+prior_bg
                #select
                obj_nums = torch.zeros([self.bs]).long()
                sort_idx = torch.argsort(z_pres)[:,-self.args.max_obj:]
                sort_idx_batch = torch.arange(self.bs).view(-1,1)
                z_pres_filtered = z_pres[sort_idx_batch,sort_idx]
                total_posi_num = torch.sum(z_pres_filtered>self.args.T)
                z_what_prev = torch.zeros(total_posi_num,self.args.zwd)
                z_what_h_prev = torch.zeros(total_posi_num,2*self.args.zwd)
                z_where_prev = torch.zeros(total_posi_num,4)
                z_pres_prev = torch.zeros(total_posi_num,1)
                obj_id_prev = torch.zeros(total_posi_num)
                count = 0
                for b in range(self.bs):
                    batch_idx = z_pres_filtered[b]>self.args.T
                    posi_num_b = torch.sum(batch_idx).long()
                    obj_nums[b] = posi_num_b
                    z_what_prev[count:count+posi_num_b] = z_what[b,sort_idx[b]][batch_idx]
                    z_what_h_prev[count:count+posi_num_b] = z_what_h[b,sort_idx[b]][batch_idx]
                    z_where_prev[count:count+posi_num_b] = z_where[b,sort_idx[b]][batch_idx]
                    z_pres_prev[count:count+posi_num_b] = z_pres[b,sort_idx[b]][batch_idx].view(-1,1)
                    obj_id_prev[count:count+posi_num_b] = obj_id[b,sort_idx[b]][batch_idx]
                    count+=posi_num_b
                z_where_h_prev = None
                z_pres_h_prev = None
                prior_h_prev = None
            else:
                total_obj_num = torch.sum(obj_nums)
                if total_obj_num == 0 and self.obj_count_flag != 0:
                    print("no obj propagated.")
                    self.obj_count_flag = 0
                if total_obj_num != 0 and self.obj_count_flag != 1:
                    print('total object nums: ',total_obj_num)
                    self.obj_count_flag = 1
                # tracking
                log_s,recon_tr,\
                mask_contri_tr,mask_rectified_tr,\
                z_where_tr,z_where_h_tr,\
                z_what_tr,z_what_h_tr,\
                z_pres_tr,z_pres_h_tr,\
                prior_h_tr,\
                kl_where_tr,kl_what_tr,kl_pres_tr,\
                glimpse_recon_tr,glimpse_mask_tr,obj_id_tr = self.tracker_net(img,
                                                                     img_embd_video[:,i],
                                                                     z_where_prev,
                                                                     z_where_h_prev,
                                                                     z_what_prev,
                                                                     z_what_h_prev,
                                                                     z_pres_prev,
                                                                     z_pres_h_prev,
                                                                     prior_h_prev,
                                                                     obj_nums,
                                                                     obj_id_prev)
                kl_where_tr_all += kl_where_tr
                kl_what_tr_all += kl_what_tr
                kl_pres_tr_all += kl_pres_tr
                # discover
                recon_ds,mask_contri_ds,mask_rectified_ds,\
                z_where_ds,z_what_ds,z_pres_ds,z_what_h_ds,\
                kl_where_ds,kl_what_ds,kl_pres_ds,\
                glimpse_recon_ds,glimpse_mask_ds = self.discover_net(img,
                                                                      img_embd_video[:,i],
                                                                      log_s,
                                                                      global_step)
                kl_where_ds_all += kl_where_ds
                kl_what_ds_all += kl_what_ds
                kl_pres_ds_all += kl_pres_ds
                obj_id_ds=torch.arange(16*i,16*i+16)+1
                obj_id_ds=obj_id_ds.reshape(1,16).repeat(self.bs,1)
                # get_normed_masks
                recon  = torch.cat([recon_tr,recon_ds],dim=1)
                mask_contri = torch.cat([mask_contri_tr,mask_contri_ds],dim=1)
                mask_rectified = torch.cat([mask_rectified_tr,mask_rectified_ds],dim=1)
                glimpse_recon = torch.cat([glimpse_recon_tr,glimpse_recon_ds],dim=1)
                glimpse_mask = torch.cat([glimpse_mask_tr,glimpse_mask_ds],dim=1)
                z_where = torch.cat([z_where_tr,z_where_ds],dim=1)
                z_what = torch.cat([z_what_tr,z_what_ds],dim=1)
                z_pres = torch.cat([z_pres_tr,z_pres_ds],dim=1)
                z_where_h = torch.cat([z_where_h_tr,torch.zeros([self.bs,16,64])],dim=1)
                z_what_h = torch.cat([z_what_h_tr,z_what_h_ds],dim=1)
                z_pres_h = torch.cat([z_pres_h_tr,torch.zeros([self.bs,16,64])],dim=1)
                prior_h = torch.cat([prior_h_tr,torch.zeros([self.bs,16,64])],dim=1)
                mask_entropy_loss,mask,log_s1 = self.get_normed_masks(mask_contri,mask_rectified)
                obj_id=torch.cat([obj_id_tr.long(),obj_id_ds],dim=1)
                log_dict = self.logging(recon,mask,z_where,z_what,z_pres,glimpse_recon,glimpse_mask,obj_id)
                self.logger.append(log_dict)
                # background
                h_bg,prior_bg,recon_bg = self.background_net(img,
                                                             log_s1,
                                                             h_bg,
                                                             self.logger)
                kl_what_bg_all+=prior_bg
                # loss
                bg_recon_loss = self.get_recon_loss_bg(img,recon_bg,log_s1.exp())
                recon_loss = self.get_recon_loss(img,recon,mask,recon_bg,log_s1.exp())
                prior_loss = kl_where_ds+kl_what_ds+kl_pres_ds+kl_where_tr+kl_what_tr+kl_pres_tr+prior_bg
                #select
                obj_nums = torch.zeros([self.bs]).long()
                sort_idx = torch.argsort(z_pres)[:,-self.args.max_obj:]
                sort_idx_batch = torch.arange(self.bs).view(-1,1)
                z_pres_filtered = z_pres[sort_idx_batch,sort_idx]
                total_posi_num = torch.sum(z_pres_filtered>self.args.T)
                z_what_prev = torch.zeros(total_posi_num,self.args.zwd)
                z_where_prev = torch.zeros(total_posi_num,4)
                z_pres_prev = torch.zeros(total_posi_num,1)
                z_what_h_prev = torch.zeros(total_posi_num,2*self.args.zwd)
                z_where_h_prev = torch.zeros(total_posi_num,64)
                z_pres_h_prev = torch.zeros(total_posi_num,64)
                prior_h_prev = torch.zeros(total_posi_num,64)
                obj_id_prev = torch.zeros(total_posi_num)
                count = 0
                for b in range(self.bs):
                    batch_idx = z_pres_filtered[b]>self.args.T
                    posi_num_b = torch.sum(batch_idx).long()
                    obj_nums[b] = posi_num_b
                    z_what_prev[count:count+posi_num_b] = z_what[b,sort_idx[b]][batch_idx]
                    z_where_prev[count:count+posi_num_b] = z_where[b,sort_idx[b]][batch_idx]
                    z_pres_prev[count:count+posi_num_b] = z_pres[b,sort_idx[b]][batch_idx].view(-1,1)
                    z_what_h_prev[count:count+posi_num_b] = z_what_h[b,sort_idx[b]][batch_idx]
                    z_where_h_prev[count:count+posi_num_b] = z_where_h[b,sort_idx[b]][batch_idx]
                    z_pres_h_prev[count:count+posi_num_b] = z_pres_h[b,sort_idx[b]][batch_idx]
                    prior_h_prev[count:count+posi_num_b] = prior_h[b,sort_idx[b]][batch_idx]
                    obj_id_prev[count:count+posi_num_b] = obj_id[b,sort_idx[b]][batch_idx]
                    count+=posi_num_b
            #loss
            recon_loss_all+=recon_loss
            prior_loss_all+=prior_loss
            mask_entropy_loss_all+=mask_entropy_loss
        mask_entropy_loss_all=mask_entropy_loss_all*self.args.ew
        #collect loss
        loss_log = AttrDict()
        loss_log.recon_loss = recon_loss_all/float(self.args.L)
        loss_log.prior_loss = prior_loss_all/float(self.args.L)
        loss_log.mask_entropy_loss = mask_entropy_loss_all/float(self.args.L)
        loss_log.kl_where_ds=kl_where_ds_all/float(self.args.L)
        loss_log.kl_what_ds=kl_what_ds_all/float(self.args.L)
        loss_log.kl_pres_ds=kl_pres_ds_all/float(self.args.L)
        loss_log.kl_where_tr=kl_where_tr_all/float(self.args.L)
        loss_log.kl_what_tr=kl_what_tr_all/float(self.args.L)
        loss_log.kl_pres_tr=kl_pres_tr_all/float(self.args.L)
        loss_log.kl_what_bg=kl_what_bg_all/float(self.args.L)
        loss_log.bg_loss=prior_bg/float(self.args.L)+bg_recon_loss/float(self.args.L)
        loss_log.loss = recon_loss_all/float(self.args.L)+prior_loss_all/float(self.args.L)+mask_entropy_loss_all/float(self.args.L)
        return loss_log

    def get_normed_masks(self,mask_contri,mask_rectified):
        #mask_contri: bs,obj_num,1,128,128
        obj_num = mask_contri.shape[1]
        mask_share = F.softmax(mask_rectified,dim=1)
        #mask_share: bs,obj_num,1,128,128
        mask_joint = torch.tanh(torch.sum(mask_contri,dim=1,keepdim=True))
        #mask_joint: bs,1,1,128,128
        mask_normed = mask_share*mask_joint.repeat(1,obj_num,1,1,1)
        #mask_normed: bs,obj_num,1,128,128
        log_s1 = torch.log(1. - mask_joint + 1e-5).view(-1,1,self.img_size,self.img_size)
        #log_s1: bs,1,128,128
        entropy_loss = -torch.sum(mask_share*torch.log(mask_share+1e-5),dim=1,keepdim=True)*mask_joint
        return torch.sum(entropy_loss),mask_normed,log_s1

    def get_recon_loss_bg(self,img,recon_bg,mask_bg):
        p_bg = Normal(recon_bg,self.bg_std)
        lp_bg = p_bg.log_prob(img).view(-1,1,3,self.img_size,self.img_size)
        log_mx = torch.log(mask_bg+1e-5)+lp_bg
        err_ppc = -torch.log(log_mx.exp().sum(dim=1)+1e-5)
        return err_ppc.sum()

    def get_recon_loss(self,img,recon,mask,recon_bg,mask_bg):
        p_fg = Normal(recon,self.std)
        p_bg = Normal(recon_bg,self.bg_std)
        obj_num=recon.shape[1]
        lp_fg = p_fg.log_prob(img.view(-1,1,3,self.img_size,self.img_size).repeat(1,obj_num,1,1,1))
        lp_bg = p_bg.log_prob(img).view(-1,1,3,self.img_size,self.img_size)
        lp_all = torch.cat([lp_fg,lp_bg],dim=1)
        mask_all = torch.cat([mask,mask_bg.view(-1,1,1,self.img_size,self.img_size)],dim=1)
        log_mx = torch.log(mask_all+1e-5)+lp_all
        err_ppc = -torch.log(log_mx.exp().sum(dim=1)+1e-5)
        return err_ppc.sum()

    def logging(self,recon,mask,z_where,z_what,z_pres,glimpse_recon,glimpse_mask,obj_id):
        masked_recon = recon*mask
        collect_dict = AttrDict()
        collect_dict.recon = recon
        collect_dict.mask = mask
        collect_dict.masked_recon = masked_recon
        collect_dict.z_where = z_where
        collect_dict.z_what = z_what
        collect_dict.z_pres = z_pres
        collect_dict.glimpse_recon = glimpse_recon
        collect_dict.glimpse_mask = glimpse_mask
        collect_dict.obj_id = obj_id
        return collect_dict
