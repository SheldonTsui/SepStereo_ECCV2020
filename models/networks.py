import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from .sync_batchnorm import SynchronizedBatchNorm2d
import pdb

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])
        
def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))
    if(Relu):
        model.append(nn.ReLU())
    return nn.Sequential(*model)

def _get_spectrogram(mask_prediction, audio_mix):
    spec_diff_real = audio_mix[:,0,:-1,:] * mask_prediction[:,0,:,:] - audio_mix[:,1,:-1,:] * mask_prediction[:,1,:,:]
    spec_diff_img = audio_mix[:,0,:-1,:] * mask_prediction[:,1,:,:] + audio_mix[:,1,:-1,:] * mask_prediction[:,0,:,:]
    binaural_spectrogram = torch.cat((spec_diff_real.unsqueeze(1), spec_diff_img.unsqueeze(1)), 1)

    return binaural_spectrogram

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

class AVFusionBlock(nn.Module):
    def __init__(self, audio_channel, vision_channel=512):
        super().__init__()
        self.channel_mapping_conv_w = nn.Conv1d(vision_channel, audio_channel, kernel_size=1)
        self.channel_mapping_conv_b = nn.Conv1d(vision_channel, audio_channel, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, audiomap, visionmap):
        visionmap = visionmap.view(visionmap.size(0), visionmap.size(1), -1)
        vision_W = self.channel_mapping_conv_w(visionmap)
        vision_W = self.activation(vision_W)
        (bz, c, wh) = vision_W.size()

        vision_W = vision_W.view(bz, c, wh)

        vision_W = vision_W.transpose(2, 1)
        audio_size = audiomap.size()
        output = torch.bmm(vision_W, audiomap.view(bz, audio_size[1], -1)).view(bz, wh, *audio_size[2:])
        return output

class VisualNet(nn.Module):
    def __init__(self, original_resnet):
        super().__init__()
        layers = list(original_resnet.children())[0:-2]
        self.feature_extraction = nn.Sequential(*layers) #features before conv1x1

    def forward(self, x):
        x = self.feature_extraction(x)
        return x

class VisualNetDilated(nn.Module):
    def __init__(self, orig_resnet):
        super().__init__()
        from functools import partial
        fc_dim = 512
        dilate_scale = 16
        conv_size = 3

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        self.features = nn.Sequential(
            *list(orig_resnet.children())[:-2])

        self.fc = nn.Conv2d(
            512, 512, kernel_size=conv_size, padding=conv_size//2)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        return x

class AudioNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2, norm_mode='syncbn'):
        super().__init__()
        #initialize layers
        if norm_mode == 'syncbn':
            norm_layer = SynchronizedBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        self.audionet_convlayer1 = unet_conv(input_nc, ngf, norm_layer=norm_layer)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2, norm_layer=norm_layer)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4, norm_layer=norm_layer)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8, norm_layer=norm_layer)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8, norm_layer=norm_layer)
        self.audionet_upconvlayer1 = unet_upconv(1296, ngf * 8, norm_layer=norm_layer) #1296 (audio-visual feature) = 784 (visual feature) + 512 (audio feature)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 4, norm_layer=norm_layer)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2, norm_layer=norm_layer)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf, norm_layer=norm_layer)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, outermost=True, norm_layer=norm_layer)
        self.conv1x1 = create_conv(512, 8, 1, 0) #reduce dimension of extracted visual features

    def forward(self, audio_diff, audio_mix, visual_feat, return_upfeatures=False):
        audio_conv1feature = self.audionet_convlayer1(audio_mix)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        visual_feat = self.conv1x1(visual_feat)
        visual_feat = visual_feat.view(visual_feat.shape[0], -1, 1, 1) #flatten visual feature
        visual_feat = visual_feat.repeat(1, 1, audio_conv5feature.shape[-2], audio_conv5feature.shape[-1]) #tile visual feature
        
        audioVisual_feature = torch.cat((visual_feat, audio_conv5feature), dim=1)
        
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        upfeatures = [audio_upconv1feature, audio_upconv2feature, audio_upconv3feature, audio_upconv4feature]

        mask_prediction = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature), dim=1)) * 2 - 1
        binaural_spectrogram = _get_spectrogram(mask_prediction, audio_mix)
        output = {'mask_prediction': mask_prediction, 'binaural_spectrogram': binaural_spectrogram, 'audio_gt': audio_diff[:,:,:-1,:]}

        if return_upfeatures:
            return upfeatures, output
        else:
            return output 

class AssoConv(nn.Module):
    def __init__(self, ngf=64, output_nc=2, visual_feat_size=7*14, vision_channel=512, norm_mode='syncbn'):
        super().__init__()
        
        if norm_mode == 'syncbn':
            norm_layer = SynchronizedBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        self.fusion = AVFusionBlock(ngf, vision_channel)
        self.lastconv_left = unet_upconv(visual_feat_size, output_nc, outermost=True, norm_layer=norm_layer) 
        self.lastconv_right = unet_upconv(visual_feat_size, output_nc, outermost=True, norm_layer=norm_layer) 

    def forward(self, audio_mix, visual_feat, upfeatures):
        audio_upconv4feature = upfeatures[-1]
        AVfusion_feature = self.fusion(audio_upconv4feature, visual_feat)
        pred_left_mask = self.lastconv_left(AVfusion_feature) * 2 - 1
        pred_right_mask = self.lastconv_right(AVfusion_feature) * 2 - 1

        left_spectrogram = _get_spectrogram(pred_left_mask, audio_mix)
        right_spectrogram = _get_spectrogram(pred_right_mask, audio_mix)
        output = {'pred_left': left_spectrogram, 'pred_right': right_spectrogram}

        return output

class APNet(nn.Module):
    def __init__(self, ngf=64, output_nc=2, visual_feat_size=7*14, vision_channel=512, norm_mode='syncbn'):
        super().__init__()

        if norm_mode == 'syncbn':
            norm_layer = SynchronizedBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        self.fusion1 = AVFusionBlock(ngf * 8, vision_channel)
        self.fusion2 = AVFusionBlock(ngf * 4, vision_channel)
        self.fusion3 = AVFusionBlock(ngf * 2, vision_channel)
        self.fusion4 = AVFusionBlock(ngf * 1, vision_channel)

        self.fusion_upconv1 = unet_upconv(visual_feat_size, visual_feat_size, norm_layer=norm_layer)
        self.fusion_upconv2 = unet_upconv(visual_feat_size * 2, visual_feat_size, norm_layer=norm_layer)
        self.fusion_upconv3 = unet_upconv(visual_feat_size * 2, visual_feat_size, norm_layer=norm_layer)
        self.lastconv_left = unet_upconv(visual_feat_size * 2, output_nc, outermost=True, norm_layer=norm_layer)
        self.lastconv_right = unet_upconv(visual_feat_size * 2, output_nc, outermost=True, norm_layer=norm_layer)

    def forward(self, audio_mix, visual_feat, upfeatures):
        audio_upconv1feature, audio_upconv2feature, audio_upconv3feature, audio_upconv4feature = upfeatures
        AVfusion_feature1 = self.fusion1(audio_upconv1feature, visual_feat)
        AVfusion_feature1 = self.fusion_upconv1(AVfusion_feature1)
        AVfusion_feature2 = self.fusion2(audio_upconv2feature, visual_feat)
        AVfusion_feature2 = self.fusion_upconv2(torch.cat((AVfusion_feature2, AVfusion_feature1), dim=1))
        AVfusion_feature3 = self.fusion3(audio_upconv3feature, visual_feat)
        AVfusion_feature3 = self.fusion_upconv3(torch.cat((AVfusion_feature3, AVfusion_feature2), dim=1))
        AVfusion_feature4 = self.fusion4(audio_upconv4feature, visual_feat)
        AVfusion_feature4 = torch.cat((AVfusion_feature4, AVfusion_feature3), dim=1)
        
        pred_left_mask = self.lastconv_left(AVfusion_feature4) * 2 - 1
        pred_right_mask = self.lastconv_right(AVfusion_feature4) * 2 - 1
        left_spectrogram = _get_spectrogram(pred_left_mask, audio_mix)
        right_spectrogram = _get_spectrogram(pred_right_mask, audio_mix)
        output = {'pred_left': left_spectrogram, 'pred_right': right_spectrogram}

        return output

# For separation
class Rearrange(nn.Module):
    def __init__(self):
        super().__init__()
        self.poolheight = 7
        self.poolwidth = 14

    def forward(self, featl, featr):
        feat_l = F.adaptive_max_pool2d(featl, (self.poolheight, 1))
        feat_r = F.adaptive_max_pool2d(featr, (self.poolheight, 1))
        featuremap = torch.zeros(featr.size(0), featr.size(1), self.poolheight, self.poolwidth).cuda()
        featuremap[:, :, :, 0] = feat_l.squeeze(3)
        featuremap[:, :, :, -1] = feat_r.squeeze(3)
        return featuremap

