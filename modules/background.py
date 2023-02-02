import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from modules.networks import BgEncoder, BgDecoder
from torch.distributions.kl import kl_divergence as KL

class Background(nn.Module):
    def __init__(self,args):
        super(Background, self).__init__()
        self.args = args
        self.bg_encoder = BgEncoder()
        self.bg_decoder = BgDecoder()
        self.bg_prior_rnn = nn.GRUCell(16, 64)
        self.bg_prior_net = nn.Linear(64, 16 * 2)
        self.std = args.bg_std * torch.ones(1, 1, 1, 1)

    def forward(self,x, log_s,h,logger):
        bs = x.shape[0]
        mask = log_s.exp()
        z_what_mean, z_what_std = self.bg_encoder(torch.cat([x,log_s],dim=1))
        z_what_std = F.softplus(z_what_std) # bs * 20 x 64
        q_z_what = Normal(z_what_mean, z_what_std)
        z_what = q_z_what.rsample() # bs * 20 x 64
        recon = self.bg_decoder(z_what)
        y = recon * mask
        #logging
        collect_dict = logger[-1]
        collect_dict.mask_bg = mask
        collect_dict.recon_bg = recon
        collect_dict.masked_recon_bg = y
        collect_dict.z_what_bg = z_what
        # prior
        if h is None:
            p_z_what = Normal(torch.zeros([bs,16]),torch.ones([bs,16]))
            prior_loss = torch.sum(KL(p_z_what,q_z_what))*self.args.wt_bg
            h_next = torch.zeros([bs,64])
        else:
            z_what_pre = logger[-2].z_what_bg
            rnn_input = z_what_pre
            h_next = self.bg_prior_rnn(rnn_input, h)
            p_z_what_mean, p_z_what_std = self.bg_prior_net(h_next).chunk(2, -1)
            p_z_what_std = F.softplus(p_z_what_std)
            p_z_what = Normal(p_z_what_mean, p_z_what_std)
            prior_loss = torch.sum(KL(p_z_what,q_z_what))*self.args.wt_bg

        return h_next,prior_loss,recon
