import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from utils import calc_kl_z_pres_bernoulli
from torch.distributions.kl import kl_divergence as KL

from utils import spatial_transform
from attrdict import AttrDict
import pdb
class Tracker(nn.Module):

    def __init__(self, glimpse_net,args):
        super(Tracker, self).__init__()
        self.args = args
        self.att_net = nn.Sequential(nn.Conv2d(64, 64, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 4),
            nn.CELU(),
            nn.GroupNorm(8, 64)
        )
        ##### where #####
        self.where_rnn = nn.GRUCell(args.zwd+4+64, 64)
        self.z_where_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.CELU(),
            nn.Linear(64, 4 * 2)
        )
        ##### pres #####
        self.pres_rnn = nn.GRUCell(args.zwd+64+1, 64)
        self.z_pres_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.CELU(),
            nn.Linear(64, 1)
        )
        ##### what #####
        self.glimpse_net = glimpse_net
        # --- prior ---
        self.prior_rnn = nn.GRUCell(args.zwd+1+4, 64)
        self.prior_net_where = nn.Linear(64, 4 * 2)
        self.prior_net_what = nn.Linear(64, args.zwd * 2)
        self.prior_net_pres = nn.Linear(64, 1)
        # offset
        offset_y, offset_x = torch.meshgrid([torch.arange(32), torch.arange(32)])
        self.register_buffer('offset',torch.stack((offset_x, offset_y), dim=0).float())

    def forward(self, img,embd,
                z_where_prev,z_where_h_prev,
                z_what_prev,z_what_h_prev,
                z_pres_prev,z_pres_h_prev,
                prior_h_prev,obj_nums,obj_id):
        img_size=128
        bs = img.size(0)
        log_s = torch.zeros([bs,1,img_size,img_size])
        if torch.sum(obj_nums) == 0:
            return log_s,torch.zeros([bs,1,3,img_size,img_size]),\
                   torch.zeros([bs,1,1,img_size,img_size]),-1000*torch.ones([bs,1,1,img_size,img_size]),\
                   torch.zeros([bs,1,4]),torch.zeros([bs,1,64]),\
                   torch.zeros([bs,1,self.args.zwd]),torch.zeros([bs,1,2*self.args.zwd]),\
                   torch.zeros([bs,1]),torch.zeros([bs,1,64]),\
                   torch.zeros([bs,1,64]),\
                   torch.sum(torch.Tensor([0.])),torch.sum(torch.Tensor([0.])),torch.sum(torch.Tensor([0.])),\
                   torch.zeros([bs,1,3,64,64]),torch.zeros([bs,1,1,64,64]),-1*torch.ones([bs,1])
        embd_repeat_list = []
        for k,n in enumerate(obj_nums):
            if n!=0:
                embd_repeat_list.append(embd[k:k+1].repeat(n,1,1,1))
        embd_repeat = torch.cat(embd_repeat_list,dim=0)
        bb_max = torch.max(z_where_prev[:,:2],dim=1)[0].view(-1,1)
        z_where_att = z_where_prev.clone().detach()
        z_where_att[:,:2] = bb_max.repeat(1,2)
        embd_att_0 =spatial_transform(embd_repeat, z_where_att,
                                     (embd_repeat.shape[0], 64, 8, 8),
                                     inverse=True)
        embd_obj = self.att_net(embd_att_0).view(-1,64)
        #embd_obj: all_obj_num,64
        cat_embd_where = torch.cat([z_what_prev,
                                    embd_obj,
                                    z_where_prev],
                                    dim=1)
        z_where_h = self.where_rnn(cat_embd_where,z_where_h_prev)
        z_where_delta_mean,z_where_delta_std = self.z_where_net(z_where_h).chunk(2,1)
        z_where_delta_std =  F.softplus(z_where_delta_std)
        q_z_where_delta = Normal(z_where_delta_mean, z_where_delta_std)
        z_where_delta = q_z_where_delta.rsample()
        z_where_centre = z_where_prev[:,2:] + self.args.tr_sh*z_where_delta[:,2:].tanh()
        z_where_size = torch.sigmoid(z_where_delta[:,:2]) #torch.ones_like(z_where_delta[:,:2])*15 #torch.sigmoid(z_where_delta[:,:2])
        z_where = torch.cat([z_where_size,z_where_centre],dim=1)
        # get glimpse encode
        embd_att = spatial_transform(embd_repeat, z_where,
                                    (embd_repeat.shape[0], 64, 8, 8),
                                    inverse=False)

        glimpse_recon, glimpse_mask,\
        z_what_mean, z_what_std,\
        z_what,z_what_h = self.glimpse_net(embd_att,z_what_prev,z_what_h_prev)
        obj_mask = spatial_transform(torch.tanh(F.softplus(glimpse_mask)),
                                     z_where,
                                     (glimpse_mask.shape[0], 1, img_size, img_size),
                                     inverse=True)
        pixel_num=torch.sum(obj_mask,dim=[1,2,3]).detach()
        #args.zwd+1+64
        cat_embd_pres = torch.cat([z_what,embd_obj,z_pres_prev],dim=1)
        z_pres_h = self.pres_rnn(cat_embd_pres,z_pres_h_prev)
        z_pres_logits = 8.8 * torch.tanh(self.z_pres_net(z_pres_h))
        z_pres_prob = torch.sigmoid(z_pres_logits)
        z_pres_prob = z_pres_prob + (z_pres_prob.clamp(1e-5, 1 - 1e-5) - z_pres_prob).detach()
        q_z_pres = RelaxedBernoulli(temperature=0.1, probs=z_pres_prob)
        z_pres = q_z_pres.rsample()

        #prepare outputs
        n = z_pres.shape[0]
        z_pres = z_pres.view(-1,1,1,1)
        recon = spatial_transform(glimpse_recon,
                                  z_where,
                                  (n, 3, img_size, img_size),
                                  inverse=True)
        mask_contri = spatial_transform(F.softplus(glimpse_mask)*z_pres, z_where,
                                       (n, 1, img_size, img_size),inverse=True)
        mask_bias_scale = -1000.0
        mask_bias = mask_bias_scale*torch.ones([n,1,img_size,img_size])
        glimpse_mask_rectified = glimpse_mask + 100.0*z_pres
        mask_rectified = spatial_transform(glimpse_mask_rectified-mask_bias_scale,
                                           z_where,
                                           (n, 1, img_size, img_size),
                                           inverse=True) + mask_bias
        #prior
        z_pres_what_where = torch.cat([z_pres_prev,z_what_prev,z_where_prev],dim=1)
        prior_h = self.prior_rnn(z_pres_what_where,prior_h_prev)
        p_where_mean,p_where_std = self.prior_net_where(prior_h).chunk(2,1)
        p_where_std = F.softplus(p_where_std)
        p_what_mean,p_what_std = self.prior_net_what(prior_h).chunk(2,1)
        p_what_std = F.softplus(p_what_std)
        p_pres_prob = torch.sigmoid(self.prior_net_pres(prior_h)).view(-1)
        p_pres_prob = p_pres_prob + (p_pres_prob.clamp(1e-5, 1 - 1e-5) - p_pres_prob).detach()
        z_pres_prob=z_pres_prob.view(-1)
        p_z_where = Normal(p_where_mean,p_where_std)
        q_z_where = Normal(z_where_delta_mean,z_where_delta_std)
        kl_where = z_pres_prob * torch.sum(KL(p_z_where,q_z_where),dim=1)*self.args.we_tr

        p_z_what = Normal(p_what_mean,p_what_std)
        q_z_what = Normal(z_what_mean,z_what_std)
        kl_what = z_pres_prob * torch.sum(KL(p_z_what,q_z_what),dim=1)*self.args.wt_tr

        kl_pres = calc_kl_z_pres_bernoulli(z_pres_prob,p_pres_prob)*(self.args.pr_tr+0.1*pixel_num)
        #padding
        max_posi_num = torch.max(obj_nums)
        recon_p = torch.zeros([bs,max_posi_num,3,128,128])
        mask_contri_p = torch.zeros([bs,max_posi_num,1,128,128])
        mask_rectified_p = -1000*torch.ones([bs,max_posi_num,1,128,128])
        glimpse_recon_p = torch.zeros([bs,max_posi_num,3,64,64])
        glimpse_mask_p = torch.zeros([bs,max_posi_num,1,64,64])
        z_what_p = torch.zeros([bs,max_posi_num,self.args.zwd])
        z_what_h_p = torch.zeros([bs,max_posi_num,self.args.zwd*2])
        z_where_p = torch.zeros([bs,max_posi_num,4])
        z_where_h_p = torch.zeros([bs,max_posi_num,64])
        z_pres_p = torch.zeros([bs,max_posi_num])
        z_pres_h_p = torch.zeros([bs,max_posi_num,64])
        prior_h_p = torch.zeros([bs,max_posi_num,64])
        obj_id_p = -1*torch.ones([bs,max_posi_num])
        count = 0
        for b in range(bs):
            on = obj_nums[b]
            recon_p[b,:on] = recon[count:count+on]
            mask_contri_p[b,:on] = mask_contri[count:count+on]
            mask_rectified_p[b,:on] = mask_rectified[count:count+on]
            glimpse_recon_p[b,:on] = glimpse_recon[count:count+on]
            glimpse_mask_p[b,:on] = glimpse_mask[count:count+on]
            z_what_p[b,:on] = z_what[count:count+on]
            z_what_h_p[b,:on] = z_what_h[count:count+on]
            z_where_p[b,:on] = z_where[count:count+on]
            z_where_h_p[b,:on] = z_where_h[count:count+on]
            z_pres_p[b,:on] = z_pres[count:count+on].view(-1)
            z_pres_h_p[b,:on] = z_pres_h[count:count+on]
            prior_h_p[b,:on] = prior_h[count:count+on]
            obj_id_p[b,:on] = obj_id[count:count+on]
            count+=on
        log_s1 = torch.log(1.-torch.tanh(torch.sum(mask_contri_p,dim=1))+1e-5)
        return log_s1,recon_p,\
               mask_contri_p,mask_rectified_p,\
               z_where_p,z_where_h_p,\
               z_what_p,z_what_h_p,\
               z_pres_p,z_pres_h_p,\
               prior_h_p,\
               torch.sum(kl_where),torch.sum(kl_what),torch.sum(kl_pres),\
               glimpse_recon_p,glimpse_mask_p,obj_id_p
