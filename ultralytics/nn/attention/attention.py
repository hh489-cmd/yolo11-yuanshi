import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from . import MPNCOV
from . import common





#
# Channel Attention
#
class ChannelAttention(nn.Module):  # Channel-attention module
    # https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))

#
# Spatial Attention
#
class SpatialAttention(nn.Module):  # Spatial-attention module
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, 1, keepdim=True)
        max_out = torch.max(x, 1, keepdim=True)[0]
        x_cat = torch.cat([avg_out, max_out], 1)
        return x * self.act(self.cv1(x_cat))

#
# CBAM模块
#
class CBAM(nn.Module):  # Convolutional Block Attention Module
    def __init__(self, c1, c2, kernel_size=7):  # ch_in, ch_out, kernel
        super().__init__()
        self.channel_attention = ChannelAttention(c2)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))

#
# 通道Shuffle
#
def channel_shuffle(x, groups=2):
    # RESHAPE----->transpose------->Flatten
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out

#
# GAM Attention
#
class GAM_Attention(nn.Module):  # https://paperswithcode.com/paper/global-attention-mechanism-retain-information
    def __init__(self, c1, c2, group=True, rate=4):
        super(GAM_Attention, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(
                c1, c1 // rate, kernel_size=7, padding=3, groups=rate
            ) if group else nn.Conv2d(
                c1, int(c1 / rate), kernel_size=7, padding=3
            ),
            nn.BatchNorm2d(int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                c1 // rate, c2, kernel_size=7, padding=3, groups=rate
            ) if group else nn.Conv2d(
                int(c1 / rate), c2, kernel_size=7, padding=3
            ),
            nn.BatchNorm2d(c2)
        )
    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).reshape(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).reshape(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        # x_channel_att = channel_shuffle(x_channel_att, 4) #last shuffle
        x = x * x_channel_att
        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att = channel_shuffle(x_spatial_att, 4)  # last shuffle
        out = x * x_spatial_att
        # out = channel_shuffle(out, 4) # last shuffle
        return out



#
def make_model(args, parent=False):
    return SAN(args)


## non_local module
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']

        # print('Dimension: %d, mode: %s' % (dimension, mode))

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            sub_sample = nn.Upsample
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.concat_project = None
        # self.fc = nn.Linear(64,2304,bias=True)
        # self.sub_bilinear = nn.Upsample(size=(48,48),mode='bilinear')
        # self.sub_maxpool = nn.AdaptiveMaxPool2d(output_size=(48,48))
        if mode in ['embedded_gaussian', 'dot_product', 'concatenation']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode == 'embedded_gaussian':
                self.operation_function = self._embedded_gaussian
            elif mode == 'dot_product':
                self.operation_function = self._dot_product
            elif mode == 'concatenation':
                self.operation_function = self._concatenation
                self.concat_project = nn.Sequential(
                    nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
                    nn.ReLU()
                )
        elif mode == 'gaussian':
            self.operation_function = self._gaussian

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        output = self.operation_function(x)
        return output

    def _embedded_gaussian(self, x):
        batch_size, C, H, W = x.shape

        # x_sub = self.sub_bilinear(x) # bilinear downsample
        # x_sub = self.sub_maxpool(x) # maxpool downsample

        ##
        # g_x = x.view(batch_size, self.inter_channels, -1)
        # g_x = g_x.permute(0, 2, 1)
        #
        # # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
        # theta_x = x.view(batch_size, self.inter_channels, -1)
        # theta_x = theta_x.permute(0, 2, 1)
        # fc = self.fc(theta_x)
        # # phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        # # f = torch.matmul(theta_x, phi_x)
        # # return f
        # # f_div_C = F.softmax(fc, dim=-1)
        # return fc

        ##
        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        # return f
        f_div_C = F.softmax(f, dim=-1)
        # return f_div_C
        # (b, thw, thw)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _dot_product(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _concatenation(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # (b, c, N, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        # (b, c, 1, N)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)

        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            # nn.Sigmoid()
            # nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        _, _, h, w = x.shape
        y_ave = self.avg_pool(x)
        # y_max = self.max_pool(x)
        y_ave = self.conv_du(y_ave)
        # y_max = self.conv_du(y_max)
        # y = y_ave + y_max
        # expand y to C*H*W
        # expand_y = y.expand(-1,-1,h,w)
        return y_ave


class SOCA(nn.Module):
    def __init__(self, channel, c2=None,reduction=1):
        super(SOCA, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):  # , hr_hea):
        batch_size, C, h, w = x.shape  # x: NxCxHxW
        N = int(h * w)
        min_h = min(h, w)
        h1 = 64
        w1 = 64
        if h < h1 and w < w1:
            x_sub = x
        elif h < h1 and w > w1:
            # H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, :, W:(W + w1)]
        elif w < w1 and h > h1:
            H = (h - h1) // 2
            # W = (w - w1) // 2
            x_sub = x[:, :, H:H + h1, :]
        else:
            H = (h - h1) // 2
            W = (w - w1) // 2
            x_sub = x[:, :, H:(H + h1), W:(W + w1)]
        cov_mat = MPNCOV.CovpoolLayer(x_sub)  # Global Covariance pooling layer
        cov_mat_sqrt = MPNCOV.SqrtmLayer(cov_mat,
                                  5)  # Matrix square root layer( including pre-norm,Newton-Schulz iter. and post-com. with 5 iteration)
        ##
        cov_mat_sum = torch.mean(cov_mat_sqrt, 1)
        cov_mat_sum = cov_mat_sum.view(batch_size, C, 1, 1)
        cov_mat_sum = cov_mat_sum.to(self.conv_du[0].weight.dtype)
        y_cov = self.conv_du(cov_mat_sum)
        return y_cov #*x


## self-attention+ channel attention module
class Nonlocal_CA(nn.Module):
    def __init__(self, in_feat=64, inter_feat=32, reduction=8, sub_sample=False, bn_layer=True):
        super(Nonlocal_CA, self).__init__()
        # second-order channel attention
        self.soca = SOCA(in_feat, reduction=reduction)
        # nonlocal module
        self.non_local = (
            NONLocalBlock2D(in_channels=in_feat, inter_channels=inter_feat, sub_sample=sub_sample, bn_layer=bn_layer))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ## divide feature map into 4 part
        batch_size, C, H, W = x.shape
        H1 = int(H / 2)
        W1 = int(W / 2)
        nonlocal_feat = torch.zeros_like(x)

        feat_sub_lu = x[:, :, :H1, :W1]
        feat_sub_ld = x[:, :, H1:, :W1]
        feat_sub_ru = x[:, :, :H1, W1:]
        feat_sub_rd = x[:, :, H1:, W1:]

        nonlocal_lu = self.non_local(feat_sub_lu)
        nonlocal_ld = self.non_local(feat_sub_ld)
        nonlocal_ru = self.non_local(feat_sub_ru)
        nonlocal_rd = self.non_local(feat_sub_rd)
        nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
        nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
        nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
        nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd

        return nonlocal_feat


## Residual  Block (RB)
class RB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(inplace=True),
                 res_scale=1, dilation=2):
        super(RB, self).__init__()
        modules_body = []

        # self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma1 = 1.0
        # self.salayer = SALayer(n_feat, reduction=reduction, dilation=dilation)
        # self.salayer = SALayer2(n_feat, reduction=reduction, dilation=dilation)

        self.conv_first = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias),
                                        act,
                                        conv(n_feat, n_feat, kernel_size, bias=bias)
                                        )

        self.res_scale = res_scale

    def forward(self, x):
        y = self.conv_first(x)
        y = y + x

        return y


## Local-source Residual Attention Group (LSRARG)
class LSRAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(LSRAG, self).__init__()
        ##
        self.rcab = nn.ModuleList([RB(conv, n_feat, kernel_size, reduction, \
                                      bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1) for _ in
                                   range(n_resblocks)])
        self.soca = (SOCA(n_feat, reduction=reduction))
        self.conv_last = (conv(n_feat, n_feat, kernel_size))
        self.n_resblocks = n_resblocks
        ##
        # modules_body = []
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma = 0.2
        # for i in range(n_resblocks):
        #     modules_body.append(RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1))
        # modules_body.append(SOCA(n_feat,reduction=reduction))
        # # modules_body.append(Nonlocal_CA(in_feat=n_feat, inter_feat=n_feat//8, reduction =reduction, sub_sample=False, bn_layer=False))
        # modules_body.append(conv(n_feat, n_feat, kernel_size))
        # self.body = nn.Sequential(*modules_body)
        ##

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)
        return nn.ModuleList(layers)
        # return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        # batch_size,C,H,W = x.shape
        # y_pre = self.body(x)
        # y_pre = y_pre + x
        # return y_pre

        ## share-source skip connection

        for i, l in enumerate(self.rcab):
            # x = l(x) + self.gamma*residual
            x = l(x)
        x = self.soca(x)
        x = self.conv_last(x)

        x = x + residual

        return x
        ##


## Second-order Channel Attention Network (SAN)
class SAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(SAN, self).__init__()
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(inplace=True)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        # self.soca= SOCA(n_feats, reduction=reduction)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        ## share-source skip connection

        ##
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma = 0.2
        self.n_resgroups = n_resgroups
        self.RG = nn.ModuleList([LSRAG(conv, n_feats, kernel_size, reduction, \
                                       act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) for _ in
                                 range(n_resgroups)])
        self.conv_last = conv(n_feats, n_feats, kernel_size)

        # modules_body = [
        #     ResidualGroup(
        #         conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
        #     for _ in range(n_resgroups)]
        # modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.non_local = Nonlocal_CA(in_feat=n_feats, inter_feat=n_feats // 8, reduction=8, sub_sample=False,
                                     bn_layer=False)

        self.head = nn.Sequential(*modules_head)
        # self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block)

        return nn.ModuleList(layers)
        # return nn.Sequential(*layers)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        ## add nonlocal
        xx = self.non_local(x)

        # share-source skip connection
        residual = xx

        # res = self.RG(xx)
        # res = res + xx
        ## share-source residual gruop
        for i, l in enumerate(self.RG):
            xx = l(xx) + self.gamma * residual
            # xx = self.gamma*xx + residual
        # body part
        # res = self.body(xx)
        ##
        ## add nonlocal
        res = self.non_local(xx)
        ##
        # res = self.soca(res)
        # res += x
        res = res + x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))