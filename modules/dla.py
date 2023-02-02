import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
class ImgEncoder(nn.Module):
    def __init__(self,c):
        super(ImgEncoder, self).__init__()
        self.l0 = nn.Sequential(
            nn.Conv2d(c, 16, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16))
        self.l1 = nn.Sequential(
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 32))
        self.l2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64))
        self.l3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128))
        self.l4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(32, 256),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(256//8, 256))

    def forward(self, x):
        x1 = self.l0(x)
        x2 = self.l1(x1)
        x3 = self.l2(x2)
        x4 = self.l3(x3)
        x5 = self.l4(x4)

        return [x1,x2,x3,x4,x5]

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

class DLA(nn.Module):
    def __init__(self,c):
        super(DLA, self).__init__()
        self.base = ImgEncoder(c)
        #####
        self.proj_0 = DeformConv(256, 128)
        self.node_0 = DeformConv(128, 128)

        self.up_0 = nn.ConvTranspose2d(128, 128, 2 * 2, stride=2,
                                       padding=1, output_padding=0,
                                       groups=128, bias=False)
        fill_up_weights(self.up_0)
        #####
        self.proj_10 = DeformConv(128, 64)
        self.node_10 = DeformConv(64, 64)

        self.up_10 = nn.ConvTranspose2d(64, 64, 2 * 2, stride=2,
                                   padding=1, output_padding=0,
                                   groups=64, bias=False)
        fill_up_weights(self.up_10)

        self.proj_11 = DeformConv(128, 64)
        self.node_11 = DeformConv(64, 64)

        self.up_11 = nn.ConvTranspose2d(64, 64, 2 * 2, stride=2,
                                   padding=1, output_padding=0,
                                   groups=64, bias=False)
        fill_up_weights(self.up_11)
        #####
        self.proj_20 = DeformConv(64, 32)
        self.node_20 = DeformConv(32, 32)

        self.up_20 = nn.ConvTranspose2d(32, 32, 2 * 2, stride=2,
                                   padding=1, output_padding=0,
                                   groups=32, bias=False)
        fill_up_weights(self.up_20)

        self.proj_21 = DeformConv(64, 32)
        self.node_21 = DeformConv(32, 32)

        self.up_21 = nn.ConvTranspose2d(32, 32, 2 * 2, stride=2,
                                   padding=1, output_padding=0,
                                   groups=32, bias=False)
        fill_up_weights(self.up_21)

        self.proj_22 = DeformConv(64, 32)
        self.node_22 = DeformConv(32, 32)

        self.up_22 = nn.ConvTranspose2d(32, 32, 2 * 2, stride=2,
                                   padding=1, output_padding=0,
                                   groups=32, bias=False)
        fill_up_weights(self.up_22)
        #####
        self.proj_30 = DeformConv(64, 32)
        self.node_30 = DeformConv(32, 32)

        self.up_30 = nn.ConvTranspose2d(32, 32, 2 * 2, stride=2,
                                   padding=1, output_padding=0,
                                   groups=32, bias=False)
        fill_up_weights(self.up_30)

        self.proj_31 = DeformConv(128, 32)
        self.node_31 = DeformConv(32, 32)

        self.up_31 = nn.ConvTranspose2d(32, 32, 4 * 2, stride=4,
                                   padding=2, output_padding=0,
                                   groups=32, bias=False)
        fill_up_weights(self.up_31)

        self.proj_32 = DeformConv(256, 32)
        self.node_32 = DeformConv(32, 32)

        self.up_32 = nn.ConvTranspose2d(32, 32, 8 * 2, stride=8,
                                   padding=4, output_padding=0,
                                   groups=32, bias=False)
        fill_up_weights(self.up_32)

    def forward(self, x):
        layers = self.base(x)
        #layer_0: torch.Size([2, 16, 64, 64])
        #layer_1: torch.Size([2, 32, 32, 32])
        #layer_2: torch.Size([2, 64, 16, 16])
        #layer_3: torch.Size([2, 128, 8, 8])
        #layer_4: torch.Size([2, 256, 4, 4])
        out = [layers[-1]] # start with 32
        #
        layers[4] = self.up_0(self.proj_0(layers[4]))#layer_4: 2,128,8,8
        layers[4] = self.node_0(layers[4] + layers[3])#layer_4: 2,128,8,8
        out.insert(0,layers[-1])
        #
        layers[3] = self.up_10(self.proj_10(layers[3]))#layer_3:2,64,16,16
        layers[3] = self.node_10(layers[3] + layers[2])#layer_3:2,64,16,16

        layers[4] = self.up_11(self.proj_11(layers[4]))#layer_4: 2,64,16,16
        layers[4] = self.node_11(layers[4] + layers[3])#layer_4: 2,64,16,16
        out.insert(0,layers[-1])
        #
        layers[2] = self.up_20(self.proj_20(layers[2]))#layer_2:2,32,32,32
        layers[2] = self.node_20(layers[2] + layers[1])#layer_3:2,64,16,16

        layers[3] = self.up_21(self.proj_21(layers[3]))#layer_3:2,32,32,32
        layers[3] = self.node_21(layers[3] + layers[2])#layer_3:2,32,32,32

        layers[4] = self.up_22(self.proj_22(layers[4]))#layer_4: 2,32,32,32
        layers[4] = self.node_22(layers[4] + layers[3])#layer_4: 2,32,32,32
        out.insert(0,layers[-1])
        #y_0: torch.Size([2, 32, 32, 32])
        #y_1: torch.Size([2, 64, 16, 16])
        #y_2: torch.Size([2, 128, 8, 8])
        #y_3: torch.Size([2, 256, 4, 4])
        y = []
        for i in range(4):
            y.append(out[i].clone())
        y[1] = self.up_30(self.proj_30(y[1]))#y1: 2,32,32,32
        y[1] = self.node_30(y[0] + y[1])#y1: 2,32,32,32

        y[2] = self.up_31(self.proj_31(y[2]))#y2: 2,64,16,16
        y[2] = self.node_31(y[2] + y[1])#y2: 2,64,16,16

        y[3] = self.up_32(self.proj_32(y[3]))#y2: 2,64,16,16
        y[3] = self.node_32(y[3] + y[2])#y2: 2,64,16,16

        return y[3]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Conv2d(chi,
                              cho,
                              kernel_size=(3,3),
                              stride=1,
                              padding=1,
                              dilation=1,
                              bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size=3, map_h=8, map_w=8):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv_x = nn.Conv2d(in_channels=self.input_dim,
                                out_channels=4 * self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=True)

        self.conv_h = nn.Conv2d(in_channels=self.hidden_dim,
                                out_channels=4 * self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding=self.padding,
                                bias=False)

        self.Wci = nn.Parameter(torch.zeros(1, self.hidden_dim, map_h, map_w),
                                requires_grad=True)
        self.Wcf = nn.Parameter(torch.zeros(1, self.hidden_dim, map_h, map_w),
                                requires_grad=True)
        self.Wco = nn.Parameter(torch.zeros(1, self.hidden_dim, map_h, map_w),
                                requires_grad=True)

        self.register_buffer('h_0', torch.zeros(1, self.hidden_dim, 1, 1))
        self.register_buffer('c_0', torch.zeros(1, self.hidden_dim, 1, 1))

    def forward(self, x, h_c):
        h_cur, c_cur = h_c

        xi, xf, xo, xc = self.conv_x(x).split(self.hidden_dim, dim=1)

        hi, hf, ho, hc = self.conv_h(h_cur).split(self.hidden_dim, dim=1)

        # print(xi.shape)
        # print(hi.shape)
        # print(c_cur.shape)
        # print(self.Wci.shape)

        i = torch.sigmoid(xi + hi + c_cur * self.Wci)
        f = torch.sigmoid(xf + hf + c_cur * self.Wcf)
        c_next = f * c_cur + i * torch.tanh(xc + hc)
        o = torch.sigmoid(xo + ho + c_cur * self.Wco)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, inp_size):
        return self.h_0.expand(batch_size, -1, inp_size[-2], inp_size[-1]), \
               self.c_0.expand(batch_size, -1, inp_size[-2], inp_size[-1])

class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, map_h, map_w):
        super(ConvLSTM, self).__init__()
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          map_h=map_h, map_w=map_w))

        self.cell_list = nn.ModuleList(cell_list)

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def forward(self, x):
        bs = x.size(0)
        layer_output_h_list = []
        layer_log_list = []

        seq_len = x.size(1)
        cur_layer_input = x

        for layer_idx in range(self.num_layers):

            output_h = []
            for t in range(seq_len):
                if t == 0:
                    h, c = self.cell_list[layer_idx].init_hidden(bs, x.size())
                h, c = self.cell_list[layer_idx](x=cur_layer_input[:, t, :, :, :],
                                                      h_c=[h, c])
                output_h.append(h)

            layer_output_h = torch.stack(output_h, dim=1)
            cur_layer_input = layer_output_h

            layer_output_h_list.append(layer_output_h)

        return layer_output_h_list[-1]

class ConvLSTMEncoder(nn.Module):

    def __init__(self,c):
        super(ConvLSTMEncoder, self).__init__()
        self.image_enc = DLA(c)
        self.conv_lstm = ConvLSTM(input_dim=32, hidden_dim=[64, 64],
                                  kernel_size=(3, 3), num_layers=2, map_h=32,
                                  map_w=32)

    def forward(self, x):
        """

        :param x: (bs, T, dim, cell_h, cell_w)
        :return:
        """
        bs = x.size(0)
        img_conv_enc = self.image_enc(x.view(-1, x.size(2), x.size(3), x.size(4)))

        img_conv_enc = img_conv_enc.view(bs, -1, img_conv_enc.size(-3), img_conv_enc.size(-2), img_conv_enc.size(-1))

        img_enc = self.conv_lstm(img_conv_enc)

        return img_enc

