import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
####################################  MLP   ####################################
class MLP(nn.Module):
    """Two-layer fully-connected ELU net."""

    def __init__(self, n_in, n_hid, n_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)

    def forward(self, inputs):
        x = F.elu(self.fc1(inputs))
        x = self.fc2(x)
        return x
####################################  Bg NET   ####################################
class BgDecoder(nn.Module):

    def __init__(self):
        super(BgDecoder, self).__init__()
        self.dec = nn.Sequential(
            nn.Conv2d(16, 256, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),

            nn.Conv2d(256, 256 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),

            nn.Conv2d(256, 128 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),

            nn.Conv2d(128, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),

            nn.Conv2d(64, 16 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
        )

        self.bg_dec = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 3, 3, 1, 1)
        )

    def forward(self, x):
        o = self.dec(x.view(-1, 16, 1, 1))
        bg = torch.sigmoid(self.bg_dec(o))
        return bg

class BgEncoder(nn.Module):

    def __init__(self):
        super(BgEncoder, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(4, 16, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 16 * 2, 4),
            # nn.MaxPool2d(3, stride=1)
        )

    def forward(self, x):
        bs = x.size(0)
        bg_what_mean, bg_what_std = self.enc(x).view(bs, -1).chunk(2, -1)

        return bg_what_mean, bg_what_std
#################################### Glimpse Net ####################################
#receptive field 150
class GlimpseNet(nn.Module):

    def __init__(self,args):
        super(GlimpseNet, self).__init__()
        self.enc_cnn = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, 4),
            nn.CELU(),
            nn.GroupNorm(16, 128),
        )
        self.rnn = nn.GRUCell(128+args.zwd,args.zwd*2)
        self.enc_what_rnn = nn.Linear(args.zwd*2, args.zwd * 2)

        self.dec = nn.Sequential(
            nn.Conv2d(args.zwd, 128, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),

            nn.Conv2d(128, 64 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),

            nn.Conv2d(64, 64 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),

            nn.Conv2d(64, 32 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 32),

            nn.Conv2d(32, 16 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(4, 16),

            nn.Conv2d(16, 8, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 8),
            nn.Conv2d(8, 4, kernel_size=1)
        )

    def forward(self, x,z_prev=None,h=None):
        x = self.enc_cnn(x)
        h_next = self.rnn(torch.cat([x.flatten(start_dim=1),z_prev],dim=1),h)
        z_what_mean, z_what_std = self.enc_what_rnn(h_next).chunk(2, -1)

        z_what_std = F.softplus(z_what_std)
        q_z_what = Normal(z_what_mean, z_what_std)
        z_what_delta = q_z_what.rsample()
        z_what = z_prev+z_what_delta

        logits = self.dec(z_what.view(z_what.size(0), -1, 1, 1))
        glimpse_recon = torch.sigmoid(logits[:,:3])
        glimpse_mask = 50.0*torch.tanh(logits[:,-1:])
        return glimpse_recon, glimpse_mask,\
               z_what_mean, z_what_std,\
               z_what,h_next
