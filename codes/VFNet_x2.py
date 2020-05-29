import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from dcn.deform_conv import ModulatedDeformConvPack as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

class TempConv(nn.Module):
    '''
    ---conv3d-bn-elu---
    '''
    def __init__(self, c_in, c_out, k_size=(1,3,3), stride=(1,1,1),padding=(0,1,1)):
        super(TempConv, self).__init__()
        self.conv3d = nn.Conv3d(c_in,c_out,kernel_size=k_size,stride=stride,padding=padding)
        self.bn = nn.BatchNorm3d(c_out)
    def forward(self,x):
        return F.elu(self.bn(self.conv3d(x)), inplace=False)
class Upscale(TempConv):
    def __init__(self, c_in, c_out, k_size=(3,3,3), stride=(1,1,1),padding=(1,1,1),scale_factor=(1,2,2)):
        super(Upscale,self).__init__(c_in,c_out)
        self.scale_factor = scale_factor
    def forward(self,x):
        return F.elu(self.bn(self.conv3d(F.interpolate(x,
                                                       scale_factor=self.scale_factor,
                                                       mode="trilinear",
                                                       align_corners=False))),
                     inplace=False)
class MaskNetwork(nn.Module):
    def __init__(self, nf1=64, nf2=128, nf3=256):
        super(MaskNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.ReplicationPad3d((1,1,1,1,1,1)),
            TempConv(3,nf1,k_size=(3,3,3), stride=(1,2,2), padding=(0,0,0)),#H/2,W/2
            TempConv(nf1,nf2,k_size=(3,3,3), padding=(1,1,1)),
            TempConv(nf2,nf2,k_size=(3,3,3), padding=(1,1,1)),
            TempConv(nf2,nf3,k_size=(3,3,3), stride=(1,2,2), padding=(1,1,1)),#H/4,W/4
            TempConv(nf3,nf3,k_size=(3,3,3), padding=(1,1,1)),
            TempConv(nf3,nf3,k_size=(3,3,3), padding=(1,1,1)),
            TempConv(nf3,nf3,k_size=(3,3,3), padding=(1,1,1)),
            TempConv(nf3,nf3,k_size=(3,3,3), padding=(1,1,1)),
            Upscale(nf3,nf2),#H/2,W/2
            TempConv(nf2,nf1,k_size=(3,3,3), padding=(1,1,1)),
            TempConv(nf1,nf1,k_size=(3,3,3), padding=(1,1,1)),
            Upscale(nf1,nf1),#H,W
            nn.Conv3d(nf1, nf1, kernel_size=(3,3,3), stride=(1,1,1),padding=(1,1,1))
        )
    def forward(self,x):
        B,N,C,H,W = x.size()
        y = torch.tanh(self.layers(x))
        y = y.permute(0,2,1,3,4).contiguous()
        ## 混合时间和通道两个维度
        return y.view(B,-1,H,W)
class ResidualBlock_noBN(nn.Module):
    '''
    ---conv-relu-conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN,self).__init__()
        self.conv1 = nn.Conv2d(nf,nf,3,1,1,bias=True)
        self.conv2 = nn.Conv2d(nf,nf,3,1,1,bias=True)
        # initialization
        nn.init.kaiming_normal_(self.conv1.weight, a=0, mode="fan_in")
        self.conv1.bias.data.zero_()
        nn.init.kaiming_normal_(self.conv2.weight, a=0, mode="fan_in")
        self.conv2.bias.data.zero_()
    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return x+y

class PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''
    def __init__(self, nf=64, groups=8):
        super(PCD_Align,self).__init__()
        #L3: level3, 1/4 spatial size
        self.L3_offest_conv1 = nn.Conv2d(nf*2, nf,3,1,1,bias=True)#concat for diff
        self.L3_offest_conv2 = nn.Conv2d(nf,nf,3,1,1,bias=True)
        self.L3_dcnpack = DCN(nf,nf,3,stride=1,padding=1,dilation=1,deformable_groups=groups,
                              extra_offset_mask=True)
        #L2: level2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf*2, nf, 3,1,1,bias=True)#concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf*2, nf,3,1,1,bias=True)#concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf,nf,3,1,1,bias=True)
        self.L2_dcnpack = DCN(nf,nf,3,stride=1,padding=1,dilation=1,deformable_groups=groups,
                              extra_offset_mask=True)
        self.L2_fea_conv = nn.Conv2d(nf*2,nf,3,1,1,bias=True)#concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                              extra_offset_mask=True)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        #Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf*2,nf,3,1,1,bias=True)
        self.cas_offset_conv2 = nn.Conv2d(nf,nf,3,1,1,bias=True)
        self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups,
                               extra_offset_mask=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    def forward(self, nbr_fea_l, ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        #L3
        L3_offset = torch.cat([nbr_fea_l[2],ref_fea_l[2]],dim=1)
        L3_offset = self.lrelu(self.L3_offest_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offest_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack([nbr_fea_l[2],L3_offset]))
        #L2
        L2_offset = torch.cat([nbr_fea_l[1],ref_fea_l[1]],dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode="bilinear",align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset,L3_offset*2],dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack([nbr_fea_l[1],L2_offset])
        L3_fea = F.interpolate(L3_fea,scale_factor=2,mode="bilinear",align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea,L3_fea],dim=1)))
        # L1
        L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack([nbr_fea_l[0], L1_offset])
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        #Cascading
        offset = torch.cat([L1_fea,ref_fea_l[0]],dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack([L1_fea,offset]))

        return L1_fea

class CA_Fusion(nn.Module):
    '''Channel Attention fusion module
    '''
    def __init__(self, nf=64, nframes=5, center=2):
        super(CA_Fusion,self).__init__()
        self.center = center
        #attention
        self.Attention_conv = nn.Sequential(
            nn.Conv2d(nf*2,nf,3,1,1,bias=True),
            nn.ReLU(),
            nn.Conv2d(nf,nf,3,1,1,bias=True),
            nn.Sigmoid()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes*nf,nframes*nf,1,1,bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1,inplace=True)
    def forward(self,aligned_fea):
        B,N,C,H,W = aligned_fea.size()
        ### channel attention
        ref_fea = aligned_fea[:,self.center,:,:,:].clone()
        features = []
        for i in range(N):
            nbr_fea = aligned_fea[:,i,:,:,:].clone()
            if i!=self.center:
                ca = self.Attention_conv(torch.cat([ref_fea,nbr_fea],dim=1))
                ca = self.avg_pool(ca)
                nbr_fea *=ca
            features.append(nbr_fea)
        features = torch.cat(features,dim=1)
        ### fusion
        features = self.lrelu(self.fea_fusion(features))

        return features 
def make_layer(block, n):
    bl = []
    for i in range(n):
        bl.append(block)
    return nn.Sequential(*bl)

class VFNet(nn.Module):
    def __init__(self, nf=64, nframes=5, groups=8, front_RBs=5, back_RBs=10, center=None, upscale_factor=2):
        super(VFNet,self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center



        self.conv_first = nn.Conv2d(3, nf, 3,1,1,bias=True)
        ### mask network
        self.maskNet = MaskNetwork(nf1=nf)
        ### extrct features(for each frame)
        self.feature_extraction = make_layer(ResidualBlock_noBN(), front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf,nf,3,2,1,bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf,nf,3,1,1,bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf,nf,3,2,1,bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf,nf,3,1,1,bias=True)
        ### PCD
        self.pcd_align = PCD_Align(nf=nf, groups=groups)
        ### channel attention
        self.ca_fusion = CA_Fusion(nf=nf, nframes=nframes, center=self.center)
        ### reconstruction
        self.recon_first = nn.Conv2d(nf*nframes, nf,3,1,1,bias=True)
        self.recon_trunk = make_layer(ResidualBlock_noBN(),back_RBs)
        ### upsampling
        self.upconv1 = nn.Conv2d(nf, nf*4,3,1,1,bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_last = nn.Conv2d(64,3,3,1,1,bias=True)
        ### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1,inplace=True)
    def forward(self,x):
        B,N,C,H,W = x.size()
        xx = x.clone()
        x_center = x[:,self.center,:,:,:].contiguous()
        ### extract LR features
        # L1
        L1_fea = self.lrelu(self.conv_first(x.view(-1,C,H,W)))#B*N,C,H,W
        L1_fea = self.feature_extraction(L1_fea)
        #L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        #L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B,N,-1,H,W)
        L2_fea = L2_fea.view(B,N,-1,H//2,W//2)
        L3_fea = L3_fea.view(B,N,-1,H//4,W//4)
        ### pcd align
        #ref feature list
        ref_fea_l = [
            L1_fea[:,self.center,:,:,:].clone(),L2_fea[:,self.center,:,:,:].clone(),
            L3_fea[:,self.center,:,:,:].clone()
        ]
        aligned_fea = []
        for i in range(N):
            nbr_fea_l = [L1_fea[:,i,:,:,:].clone(),L2_fea[:,i,:,:,:].clone(),L3_fea[:,i,:,:,:].clone()]
            aligned_fea.append(self.pcd_align(nbr_fea_l,ref_fea_l))
        aligned_fea = torch.stack(aligned_fea,dim=1)#B,N,C,H,W

        fea = self.ca_fusion(aligned_fea)
        fea = fea+self.maskNet(xx.permute(0,2,1,3,4))#B,C,N,H,W
        fea = self.recon_first(fea)
        out = self.recon_trunk(fea)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.conv_last(out)
        base = F.interpolate(x_center,scale_factor=2,mode="bilinear",align_corners=False)
        out += base
        return out