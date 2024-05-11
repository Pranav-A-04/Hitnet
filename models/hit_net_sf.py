import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import math
import torchvision.models as models

def same_padding_conv(x, w, b, s):
    out_h = math.ceil(x.size(2) / s[0])
    out_w = math.ceil(x.size(3) / s[1])

    pad_h = max((out_h - 1) * s[0] + w.size(2) - x.size(2), 0)
    pad_w = max((out_w - 1) * s[1] + w.size(3) - x.size(3), 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    x = F.conv2d(x, w, b, stride=s)
    return x


class SameConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return same_padding_conv(x, self.weight, self.bias, self.stride)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        
        # Define three 1x1 convolutional layers to generate feature maps A, B, and C
        self.convA = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.convB = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.convC = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # 1x1 convolutional layer to produce the attention weight matrix
        self.conv_att = nn.Conv2d(out_channels, 1, kernel_size=1)
        
        # Sigmoid activation layer to map the output of the convolutional layer to values between 0 and 1
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Generate feature maps A, B, and C
        A = self.convA(x)
        B = self.convB(x)
        C = self.convC(x)
        
        # Combine A and B with element-wise addition
        attention = torch.add(A, B)
        
        # Apply 1x1 convolution to produce the attention weight matrix
        attention = self.conv_att(attention)
        
        # Apply sigmoid activation to get attention weights between 0 and 1
        attention = self.sigmoid(attention)
        
        # Multiply attention weights with linearly transformed input C
        O = torch.mul(attention, C)
        
        # Sum along the channel dimension to get the final output O
        #O = torch.sum(attended_C, dim=1, keepdim=True)
        
        return O
    
#class UpsampleBlock(nn.Module):
#    def __init__(self, c0, c1):
#        super().__init__()
#        self.up_conv = nn.Sequential(
#            nn.ConvTranspose2d(c0, c1, 2, 2),
#            nn.LeakyReLU(0.2),
#        )
#        self.merge_conv = nn.Sequential(
#            nn.Conv2d(c1 * 2, c1, 1),
#            nn.LeakyReLU(0.2),
#            nn.Conv2d(c1, c1, 3, 1, 1),
#            nn.LeakyReLU(0.2),
#            nn.Conv2d(c1, c1, 3, 1, 1),
#            nn.LeakyReLU(0.2),
#        )

#        #self.attn_block=AttentionBlock(c1, c1)

#    def forward(self, input, sc):
#        
#        x = self.up_conv(input)
#       # sc=self.attn_block(sc)
#        if x.size()[2:] != sc.size()[2:]:
#            x = x[:, :, : sc.size(2), : sc.size(3)]
#        x = torch.cat((x, sc), dim=1)

#        x = self.merge_conv(x)
#        return x
class LocalFeatureEnhancementModule(nn.Module):
    def __init__(self, C, initial_kernel_size=3, initial_dilation=1):
        super(LocalFeatureEnhancementModule, self).__init__()
        self.block1 = self._create_block(1, C[0], initial_kernel_size, initial_dilation)
        self.block2 = self._create_block(C[0], C[1], initial_kernel_size, initial_dilation * 2)
        self.block3 = self._create_block(C[1], C[2], initial_kernel_size, initial_dilation * 4)
        self.block4 = self._create_block(C[2], C[3], initial_kernel_size, initial_dilation * 8)

    def _create_block(self, in_channels, out_channels, kernel_size, dilation):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, dilation=dilation),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        return out1, out2, out3, out4
        
class UpsampleBlock(nn.Module):
    def __init__(self, c0, c1):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(c0, c1, 2, 2),
            nn.LeakyReLU(0.2),
        )
        self.merge_conv = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c1, c1, 3, 1, 1),		
            nn.LeakyReLU(0.2),
        )
        self.attn_block=AttentionBlock(c1, c1)

    def forward(self, input, sc):
        x = self.up_conv(input)
        sc=self.attn_block(sc)
        if x.size()[2:] != sc.size()[2:]:
            x = x[:, :, : sc.size(2), : sc.size(3)]
        x = torch.cat((x, sc), dim=1)
        x = self.merge_conv(x)
        return x


#class FeatureExtractor(nn.Module):
#    def __init__(self, C):
#        super().__init__()
#        self.down_0 = nn.Sequential(
#            nn.Conv2d(1, C[0], 3, 1, 1),
#            nn.LeakyReLU(0.2),
#        )
#        self.down_1 = nn.Sequential(
#            SameConv2d(C[0], C[1], 4, 2),
#            nn.LeakyReLU(0.2),
#            nn.Conv2d(C[1], C[1], 3, 1, 1),
#            nn.LeakyReLU(0.2),
#        )
#        self.down_2 = nn.Sequential(
#            SameConv2d(C[1], C[2], 4, 2),
#            nn.LeakyReLU(0.2),
#            nn.Conv2d(C[2], C[2], 3, 1, 1),
#            nn.LeakyReLU(0.2),
#        )
#        self.down_3 = nn.Sequential(
#            SameConv2d(C[2], C[3], 4, 2),
#            nn.LeakyReLU(0.2),
#            nn.Conv2d(C[3], C[3], 3, 1, 1),
#            nn.LeakyReLU(0.2),
#        )
#        self.down_4 = nn.Sequential(
#            SameConv2d(C[3], C[4], 4, 2),
#            nn.LeakyReLU(0.2),
#            nn.Conv2d(C[4], C[4], 3, 1, 1),
#            nn.LeakyReLU(0.2),
#            nn.Conv2d(C[4], C[4], 3, 1, 1),
#            nn.LeakyReLU(0.2),
#            nn.Conv2d(C[4], C[4], 3, 1, 1),
#            nn.LeakyReLU(0.2),
#        )
#        self.up_3 = UpsampleBlock(C[4], C[3])
#        self.up_2 = UpsampleBlock(C[3], C[2])
#        self.up_1 = UpsampleBlock(C[2], C[1])
#        self.up_0 = UpsampleBlock(C[1], C[0])

##        self.feature_enhancement = LocalFeatureEnhancementModule(C)
##        self.fuse_conv = nn.Conv2d(sum(C), C[-1], kernel_size=1)
#        #self.fuse_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
#    def forward(self, input):
#        x0 = self.down_0(input)
#        x1 = self.down_1(x0)
#        x2 = self.down_2(x1)
#        x3 = self.down_3(x2)
#        o0 = self.down_4(x3)

#        
#        o1 = self.up_3(o0, x3)
#        o2 = self.up_2(o1, x2)
#        o3 = self.up_1(o2, x1)
#        o4 = self.up_0(o3, x0)

#        #out1, out2, out3, out4 = self.feature_enhancement(input)

#        #fused_up1 = torch.cat([o1, out1], dim=1)
#        #fused_up2 = torch.cat([o2, out2], dim=1)
#        #fused_up3 = torch.cat([o3, out3], dim=1)
#        #fused_up4 = torch.cat([o4, out4], dim=1)
##        fused_up1 = torch.cat([o1, out4], dim=1)
##        fused_up2 = torch.cat([o2, out3], dim=1)
##        fused_up3 = torch.cat([o3, out2], dim=1)
##        fused_up4 = torch.cat([o4, out1], dim=1)

##        fused_up1 = self.fuse_conv(fused_up1)
##        fused_up2 = self.fuse_conv(fused_up2)
##        fused_up3 = self.fuse_conv(fused_up3)
##        fused_up4 = self.fuse_conv(fused_up4)


#        return o4, o3, o2, o1, o0



class FeatureExtractor(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.down_0 = nn.Sequential(
            nn.Conv2d(1, C[0], 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.down_1 = nn.Sequential(
            SameConv2d(C[0], C[1], 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(C[1], C[1], 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.down_2 = nn.Sequential(
            SameConv2d(C[1], C[2], 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(C[2], C[2], 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.down_3 = nn.Sequential(
            SameConv2d(C[2], C[3], 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(C[3], C[3], 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.down_4 = nn.Sequential(
            SameConv2d(C[3], C[4], 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(C[4], C[4], 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(C[4], C[4], 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(C[4], C[4], 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        
        
        self.up_3 = UpsampleBlock(C[4], C[3])
        self.up_2 = UpsampleBlock(C[3], C[2])
        self.up_1 = UpsampleBlock(C[2], C[1])
        self.up_0 = UpsampleBlock(C[1], C[0])

    def forward(self, input):
        x0 = self.down_0(input)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        o0 = self.down_4(x3)
        o1 = self.up_3(o0, x3)
        o2 = self.up_2(o1, x2)
        o3 = self.up_1(o2, x1)
        o4 = self.up_0(o3, x0)
        return o4,o3,o2,o1,o0


class FeatureExtract(nn.Module):
    def __init__(self, C):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        # Remove fully connected layer and avg pool layer
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])

        # Define upsampling blocks
        self.up_3 = self._make_upsample_block(C[4], C[3])
        self.up_2 = self._make_upsample_block(C[3],C[2])
        self.up_1 = self._make_upsample_block(C[2], C[1])
        self.up_0 = self._make_upsample_block(C[1], C[0])
        
    def _make_upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        # Downsampling (Encoder)
        x0 = self.resnet_features[:4](input)
        x1 = self.resnet_features[4](x0)
        x2 = self.resnet_features[5](x1)
        x3 = self.resnet_features[6](x2)
	
        # Upsampling (Decoder)
        o0 = self.resnet_features[6](x3)
        o1 = self.up_3(o0)
        o2 = self.up_2(o1)
        o3 = self.up_1(o2)
        o4 = self.up_0(o3)

        return o4,o3,o2,o1,o0
@functools.lru_cache()
@torch.no_grad()
def make_warp_coef(scale, device):
    center = (scale - 1) / 2
    index = torch.arange(scale, device=device) - center
    coef_y, coef_x = torch.meshgrid(index, index)
    coef_x = coef_x.reshape(1, -1, 1, 1)
    coef_y = coef_y.reshape(1, -1, 1, 1)
    return coef_x, coef_y


def disp_up(d, dx, dy, scale, tile_expand):
    n, _, h, w = d.size()
    coef_x, coef_y = make_warp_coef(scale, d.device)

    if tile_expand:
        d = d + coef_x * dx + coef_y * dy
    else:
        d = d * scale + coef_x * dx * 4 + coef_y * dy * 4

    d = d.reshape(n, 1, scale, scale, h, w)
    d = d.permute(0, 1, 4, 2, 5, 3)
    d = d.reshape(n, 1, h * scale, w * scale)
    return d


def hyp_up(hyp, scale=1, tile_scale=1):
    if scale != 1:
        d = disp_up(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], scale, tile_expand=False)
        p = F.interpolate(hyp[:, 1:], scale_factor=scale)
        hyp = torch.cat((d, p), dim=1)
    if tile_scale != 1:
        d = disp_up(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], tile_scale, tile_expand=True)
        p = F.interpolate(hyp[:, 1:], scale_factor=tile_scale)
        hyp = torch.cat((d, p), dim=1)
    return hyp


def warp_and_aggregate(hyp, left, right):
    scale = left.size(3) // hyp.size(3)
    assert scale == 4

    d_expand = disp_up(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], scale, tile_expand=True)
    d_range = torch.arange(right.size(3), device=right.device)
    d_range = d_range.view(1, 1, 1, -1) - d_expand
    d_range = d_range.repeat(1, right.size(1), 1, 1)

    cost = [torch.sum(torch.abs(left), dim=1, keepdim=True)]
    for offset in [1, 0, -1]:
        index_float = d_range + offset
        index_long = torch.floor(index_float).long()
        index_left = torch.clip(index_long, min=0, max=right.size(3) - 1)
        index_right = torch.clip(index_long + 1, min=0, max=right.size(3) - 1)
        index_weight = index_float - index_left

        right_warp_left = torch.gather(right, dim=-1, index=index_left.long())
        right_warp_right = torch.gather(right, dim=-1, index=index_right.long())
        right_warp = right_warp_left + index_weight * (
            right_warp_right - right_warp_left
        )
        cost.append(torch.sum(torch.abs(left - right_warp), dim=1, keepdim=True))
    cost = torch.cat(cost, dim=1)

    n, c, h, w = cost.size()
    cost = cost.reshape(n, c, h // scale, scale, w // scale, scale)
    cost = cost.permute(0, 3, 5, 1, 2, 4)
    cost = cost.reshape(n, scale * scale * c, h // scale, w // scale)
    return cost


class ResBlock(nn.Module):
    def __init__(self, c0, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c0, c0, 3, 1, dilation, dilation),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c0, c0, 3, 1, dilation, dilation),
        )
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, input):
        x = self.conv(input)
        x = x + input
        x = self.relu(x)
        return x


def make_cost_volume_v2(left, right, max_disp):
    d_range = torch.arange(max_disp, device=left.device)
    d_range = d_range.view(1, 1, -1, 1, 1)

    x_index = torch.arange(left.size(3), device=left.device)
    x_index = x_index.view(1, 1, 1, 1, -1)

    x_index = torch.clip(4 * x_index - d_range + 1, 0, right.size(3) - 1).repeat(
        right.size(0), right.size(1), 1, right.size(2), 1
    )
    right = torch.gather(
        right.unsqueeze(2).repeat(1, 1, max_disp, 1, 1), dim=-1, index=x_index
    )

    return left.unsqueeze(2) - right


class InitDispNet(nn.Module):
    def __init__(self, cin, cout, cf=None):
        super().__init__()
        self.conv_em = nn.Conv2d(cin, cout, 4)
        self.relu_conv = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(cout, cout, 1),
            nn.LeakyReLU(0.2),
        )
        if cf is None:
            cf = cout
        self.tile_feautre = nn.Sequential(
            nn.Conv2d(cf + 1, 13, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, feature_left, feature_right, max_disp, feature_ref=None):
        feature_left_tilde = F.conv2d(
            feature_left,
            self.conv_em.weight,
            self.conv_em.bias,
            stride=(4, 4),
        )
        feature_left_tilde = self.relu_conv(feature_left_tilde)

        feature_right_tilde = same_padding_conv(
            feature_right,
            self.conv_em.weight,
            self.conv_em.bias,
            s=(4, 1),
        )
        feature_right_tilde = self.relu_conv(feature_right_tilde)

        cost_volume = make_cost_volume_v2(
            feature_left_tilde,
            feature_right_tilde,
            max_disp,
        )
        cost_volume = torch.norm(cost_volume, p=1, dim=1)
        cost_f, d_init = torch.min(cost_volume, dim=1, keepdim=True)
        d_init = d_init.float()

        if feature_ref is None:
            feature_ref = feature_left_tilde
        p_init = torch.cat((cost_f, feature_ref), dim=1)
        p_init = self.tile_feautre(p_init)

        return (
            torch.cat(
                (d_init, torch.zeros_like(d_init), torch.zeros_like(d_init), p_init),
                dim=1,
            ),
            cost_volume,
        )


class PropagationNet(nn.Module):
    def __init__(self, dilations, tile_size=4):
        super().__init__()
        self.conv_neighbors = nn.Sequential(
            nn.Conv2d(4 * tile_size * tile_size, 16, 1),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.res_block = []
        for d in dilations:
            self.res_block += [ResBlock(32, d)]
        self.res_block = nn.Sequential(*self.res_block)
        self.convn = nn.Conv2d(32, 17, 3, 1, 1)

    def forward_once(self, hyp, left, right):
        x = warp_and_aggregate(hyp, left, right)
        x = self.conv_neighbors(x)
        x = torch.cat((hyp, x), dim=1)
        x = self.conv1(x)
        x = self.res_block(x)
        x = self.convn(x)
        hyp = hyp + x[:, :-1, :, :]
        w = x[:, -1:, :, :]
        return hyp, w

    def forward(self, hyps, left, right):
        if len(hyps) == 1:
            hyp, w = self.forward_once(hyps[0], left, right)
        else:
            raise NotImplementedError
        return hyp


class RefinementNet(nn.Module):
    def __init__(self, cin, cres, dilations):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(cin + 16, cres, 1),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(cres, cres, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.res_block = []
        for d in dilations:
            self.res_block += [ResBlock(cres, d)]
        self.res_block = nn.Sequential(*self.res_block)
        self.convn = nn.Conv2d(cres, 16, 3, 1, 1)

    def forward(self, hpy, left):
        x = torch.cat((left, hpy), dim=1)
        x = self.conv1x1(x)
        x = self.conv1(x)
        x = self.res_block(x)
        x = self.convn(x)
        return hpy + x


class HITNet_SF(nn.Module):
    def __init__(self):
        super().__init__()
        self.align = 4
        self.max_disp = 320

        num_feature = [16, 16, 24, 24, 32]
        res_dilations = [1, 1]

        self.feature_extractor = FeatureExtractor(num_feature)
        self.init_layer_0 = InitDispNet(num_feature[0], 16, num_feature[2])
        self.prop_layer_0 = PropagationNet(res_dilations)

        res_dilations = [1, 2, 4, 8, 1, 1]
        self.refine_l0 = RefinementNet(num_feature[2], 32, res_dilations)
        self.refine_l1 = RefinementNet(num_feature[1], 32, res_dilations)
        self.refine_l2 = RefinementNet(num_feature[0], 16, res_dilations)

    def forward(self, left_img, right_img):
        n, c, h, w = left_img.size()
        w_pad = (self.align - (w % self.align)) % self.align
        h_pad = (self.align - (h % self.align)) % self.align

        left_img = F.pad(left_img, (0, w_pad, 0, h_pad))
        right_img = F.pad(right_img, (0, w_pad, 0, h_pad))

        lf = self.feature_extractor(left_img)
        rf = self.feature_extractor(right_img)

        hi_0, cv_0 = self.init_layer_0(lf[0], rf[0], self.max_disp, lf[2])
        h_0 = self.prop_layer_0([hi_0], lf[0], rf[0])
        h_1 = self.refine_l0(h_0, lf[2])
        h_2 = self.refine_l1(hyp_up(h_1, 1, 2), lf[1])
        h_3 = self.refine_l2(hyp_up(h_2, 1, 2), lf[0])[:, :, :h, :w]

        return {
            "tile_size": 4,
            "disp": h_3[:, 0:1],
            "multi_scale": [
                h_0[:, 0:1],
                h_1[:, 0:1],
                h_2[:, 0:1],
                h_3[:, 0:1],
            ],
            "cost_volume": [cv_0],
            "slant": [
                [h_0[:, 0:1], h_0[:, 1:3]],
                [h_1[:, 0:1], h_1[:, 1:3]],
                [h_2[:, 0:1], h_2[:, 1:3]],
                [h_3[:, 0:1], h_3[:, 1:3]],
            ],
            "init_disp": [hi_0[:, 0:1]],
        }


class HITNetXL_SF(nn.Module):
    def __init__(self):
        super().__init__()
        self.align = 4
        self.max_disp = 320

        num_feature = [32, 40, 48, 56, 64]
        res_dilations = [1, 1]

        self.feature_extractor = FeatureExtractor(num_feature)
        self.init_layer_0 = InitDispNet(num_feature[0], 16, num_feature[2])
        self.prop_layer_0 = PropagationNet(res_dilations)

        res_dilations = [1, 2, 4, 8, 1, 1]
        self.refine_l0 = RefinementNet(num_feature[2], 64, res_dilations)
        self.refine_l1 = RefinementNet(num_feature[1], 64, res_dilations)
        self.refine_l2 = RefinementNet(num_feature[0], 64, res_dilations)

    def forward(self, left_img, right_img):
        n, c, h, w = left_img.size()
        w_pad = (self.align - (w % self.align)) % self.align
        h_pad = (self.align - (h % self.align)) % self.align

        left_img = F.pad(left_img, (0, w_pad, 0, h_pad))
        right_img = F.pad(right_img, (0, w_pad, 0, h_pad))

        lf = self.feature_extractor(left_img)
        rf = self.feature_extractor(right_img)

        hi_0, cv_0 = self.init_layer_0(lf[0], rf[0], self.max_disp, lf[2])
        h_0 = self.prop_layer_0([hi_0], lf[0], rf[0])
        h_1 = self.refine_l0(h_0, lf[2])
        h_2 = self.refine_l1(hyp_up(h_1, 1, 2), lf[1])
        h_3 = self.refine_l2(hyp_up(h_2, 1, 2), lf[0])[:, :, :h, :w]

        return {
            "tile_size": 4,
            "disp": h_3[:, 0:1],
            "multi_scale": [
                h_0[:, 0:1],
                h_1[:, 0:1],
                h_2[:, 0:1],
                h_3[:, 0:1],
            ],
            "cost_volume": [cv_0],
            "slant": [
                [h_0[:, 0:1], h_0[:, 1:3]],
                [h_1[:, 0:1], h_1[:, 1:3]],
                [h_2[:, 0:1], h_2[:, 1:3]],
                [h_3[:, 0:1], h_3[:, 1:3]],
            ],
            "init_disp": [hi_0[:, 0:1]],
        }


if __name__ == "__main__":
    import cv2
    from thop import profile

    left = torch.rand(1, 1, 540, 960)
    right = torch.rand(1, 1, 540, 960)
    model = HITNet_SF()

    print(model(left, right)["disp"].size())

    total_ops, total_params = profile(
        model,
        (
            left,
            right,
        ),
    )
    print(
        "{:.4f} MACs(G)\t{:.4f} Params(M)".format(
            total_ops / (1000 ** 3), total_params / (1000 ** 2)
        )
    )
