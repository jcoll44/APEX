import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from utils import calc_kl_z_pres_bernoulli
from torch.distributions.kl import kl_divergence as KL
from utils import spatial_transform
from modules.networks import MLP
class Discover(nn.Module):
    def __init__(self,glimpse_net, args):
        super(Discover, self).__init__()
        self.args = args
        # z_where
        self.z_where_net = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 4 * 2, 1)
        )
        # z_pres
        self.z_pres_net = MLP(args.zwd+4+64, 64,1)
        # z_what
        self.glimpse_net = glimpse_net
        # offset
        offset_y, offset_x = torch.meshgrid([torch.arange(4), torch.arange(4)])
        self.register_buffer('offset',torch.stack((offset_x, offset_y), dim=0).float())

    def forward(self,img,embd_all,log_s,global_step):
        p_z_pres_anneal=1e-3
        img_size = 128
        bs = img.size()[0]
        n = bs * 4**2
        nc = 4**2
        #embedding
        img_embd = embd_all[:,:,[4,4,4,4,12,12,12,12,20,20,20,20,28,28,28,28],[4, 12, 20, 28]*4]
        img_embd = img_embd.view(bs,64,4,4)
        # where
        z_where_mean, z_where_std = self.z_where_net(img_embd).chunk(2, 1)
        z_where_std = F.softplus(z_where_std)
        q_z_where = Normal(z_where_mean, z_where_std)
        z_where = q_z_where.rsample()

        z_where[:, :2]=torch.sigmoid(z_where[:, :2])
        z_where[:, 2:] = 2. / 4. * (self.offset + 0.5 + 0.5*z_where[:, 2:].tanh()) - 1
        z_where = z_where.permute(0, 2, 3, 1).reshape(-1, 4)
        # what
        x_att = spatial_transform(torch.stack(nc*(embd_all,), dim=1).view(-1, 64, 32, 32),
                                  z_where,
                                  (n, 64, 8, 8),
                                  inverse=False)
        z_what_init = torch.zeros([n,self.args.zwd])
        glimpse_recon, glimpse_mask,\
        z_what_mean, z_what_std,\
        z_what,z_what_h= self.glimpse_net(x_att,z_what_init)

        obj_mask = spatial_transform(torch.tanh(F.softplus(glimpse_mask)),
                                     z_where,
                                     (n, 1, img_size, img_size),
                                     inverse=True)
        pixel_num  = torch.sum(obj_mask,dim=[1,2,3]).detach()
        # pres
        z_pres_inp = torch.cat([img_embd.permute(0, 2, 3, 1).reshape(-1,64),z_what,z_where],dim=1)
        z_pres_global = torch.sigmoid(8.8*torch.tanh(self.z_pres_net(z_pres_inp)))
        log_s_r = torch.stack(nc * (log_s,), dim=1).view(-1, 1, img_size, img_size)
        z_pres_local = torch.sum(log_s_r.exp()*obj_mask,dim=[1,2,3])/(pixel_num+1e-5)
        z_pres_prob = z_pres_global*z_pres_local.view(-1,1).detach()
        z_pres_prob = z_pres_prob + (z_pres_prob.clamp(1e-5, 1 - 1e-5) - z_pres_prob).detach()
        q_z_pres = RelaxedBernoulli(temperature=0.1, probs=z_pres_prob)
        z_pres = q_z_pres.rsample()
        # prior
        z_where_mean = z_where_mean.permute(0, 2, 3, 1).reshape(-1, 4)
        z_where_std = z_where_std.permute(0, 2, 3, 1).reshape(-1, 4)
        z_pres_prob=z_pres_prob.view(-1)

        p_z_what = Normal(torch.zeros([n,self.args.zwd]),torch.ones([n,self.args.zwd]))
        q_z_what = Normal(z_what_mean,z_what_std)
        kl_what = z_pres_prob * torch.sum(KL(p_z_what,q_z_what),dim=1)*self.args.wt_ds

        p_z_where = Normal(torch.zeros([n,4]),torch.ones([n,4]))
        q_z_where = Normal(z_where_mean,z_where_std)
        kl_where = z_pres_prob * torch.sum(KL(p_z_where,q_z_where),dim=1)*self.args.we_ds
        p_z_pres_prob = torch.ones([z_pres_prob.shape[0]])*p_z_pres_anneal
        kl_pres = calc_kl_z_pres_bernoulli(z_pres_prob,p_z_pres_prob)*(self.args.pr_ds+0.1*pixel_num)
        prior_loss = torch.sum(kl_what + kl_where + kl_pres)
        # prepare the outputs
        z_pres = z_pres.view(-1,1,1,1)
        recon = spatial_transform(glimpse_recon,
                                  z_where,
                                  (n, 3, img_size, img_size),
                                  inverse=True).view(bs,nc,3,img_size,img_size)
        mask_contri = spatial_transform(F.softplus(glimpse_mask)*z_pres, z_where,
                                       (n, 1, img_size, img_size),inverse=True).view(bs,nc,1,img_size,img_size)
        mask_bias_scale = -1000.0
        mask_bias = mask_bias_scale*torch.ones([n,1,img_size,img_size])
        glimpse_mask_rectified = glimpse_mask + 100.0*z_pres
        mask_rectified = spatial_transform(glimpse_mask_rectified-mask_bias_scale,
                                           z_where,
                                           (n, 1, img_size, img_size),
                                           inverse=True) + mask_bias
        mask_rectified = mask_rectified.view(bs,nc,1,img_size,img_size)
        z_pres = z_pres.view(bs,nc)
        z_what = z_what.view(bs,nc,self.args.zwd)
        z_what_h = z_what_h.view(bs,nc,self.args.zwd*2)
        z_where = z_where.view(bs,nc,4)
        glimpse_recon = glimpse_recon.view(bs,nc,3,64,64)
        glimpse_mask = glimpse_mask.view(bs,nc,1,64,64)
        return recon,mask_contri,mask_rectified,\
               z_where,z_what,z_pres,z_what_h,\
               torch.sum(kl_where),torch.sum(kl_what),torch.sum(kl_pres),\
               glimpse_recon,glimpse_mask
