"""
Network architectures for CycleGAN.
Based on the improved architecture from Fu (2022).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    
    Args:
        net: network to be initialized
        init_type: initialization method ('normal', 'xavier', 'kaiming', or 'orthogonal')
        init_gain: scaling factor for normal, xavier and orthogonal
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'Initialization method {init_type} is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    return net


# Improved ResNet block from Fu (2022)
class ResnetBlock(nn.Module):
    """Define a Resnet block with improved structure for style transfer"""
    def __init__(self, dim, use_bias=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, use_bias)
        
    def build_conv_block(self, dim, use_bias):
        """Construct a convolutional block with improved structure"""
        conv_block = []
        
        # First convolution with Instance Normalization
        conv_block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True)
        ]
        
        # Second convolution with Instance Normalization but no ReLU
        conv_block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            nn.InstanceNorm2d(dim)
        ]
        
        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        """Forward function with skip connection"""
        out = x + self.conv_block(x)  # Skip connection
        return out


# Improved Generator from Fu (2022)
class ResnetGenerator(nn.Module):
    """Enhanced ResNet generator architecture for style transfer"""
    def __init__(
        self, 
        input_nc=3, 
        output_nc=3, 
        ngf=64, 
        n_blocks=9, 
        use_dropout=False
    ):
        super(ResnetGenerator, self).__init__()
        
        use_bias = True  # Instance normalization doesn't have affine parameters
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]
        
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]
        
        # Resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, use_bias=use_bias)]
            
        # Upsampling with improved interpolation method
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
            
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, input):
        """Run forward pass"""
        return self.model(input)


# Multi-scale discriminator from Fu (2022)
class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator that evaluates at different resolutions"""
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_spectral_norm=True):
        super(MultiScaleDiscriminator, self).__init__()
        
        # Regular discriminator at full resolution
        self.discriminator_full = NLayerDiscriminator(
            input_nc, ndf, n_layers, use_spectral_norm=use_spectral_norm
        )
        
        # Discriminator at half resolution
        self.discriminator_half = NLayerDiscriminator(
            input_nc, ndf, n_layers, use_spectral_norm=use_spectral_norm
        )
        
        # Downsample layer for half resolution
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)
        
    def forward(self, input):
        """Return list of discriminator outputs at different scales"""
        # Full resolution
        d_full = self.discriminator_full(input)
        
        # Half resolution
        input_downsampled = self.downsample(input)
        d_half = self.discriminator_half(input_downsampled)
        
        return [d_full, d_half]


# Patch discriminator with spectral normalization option
class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator with spectral normalization option"""
    def __init__(
        self, 
        input_nc=3, 
        ndf=64, 
        n_layers=3, 
        use_spectral_norm=True,
        use_sigmoid=False
    ):
        super(NLayerDiscriminator, self).__init__()
        
        norm_layer = nn.InstanceNorm2d
        use_bias = True
        
        kw = 4
        padw = 1
        sequence = []
        
        # First layer without normalization
        if use_spectral_norm:
            sequence += [
                spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
                nn.LeakyReLU(0.2, True)
            ]
        else:
            sequence += [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]
        
        # Middle layers with normalization
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            
            if use_spectral_norm:
                sequence += [
                    spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                       kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                             kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
        
        # Last layer without downsampling
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        if use_spectral_norm:
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                   kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        else:
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        # Output layer
        if use_spectral_norm:
            sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1,
                                            kernel_size=kw, stride=1, padding=padw))]
        else:
            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
            
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
            
        self.model = nn.Sequential(*sequence)
        
    def forward(self, input):
        """Run forward pass"""
        return self.model(input)


# Perceptual feature extractor for structural loss
class PerceptualNet(nn.Module):
    """Perceptual network for extracting features for structural loss"""
    def __init__(self):
        super(PerceptualNet, self).__init__()
        # Use pre-trained VGG19 features from torchvision
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
        vgg_features = vgg.features
        
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        
        # Extract different feature levels
        for x in range(5):  # conv1_1 -> conv2_1
            self.slice1.add_module(str(x), vgg_features[x])
        for x in range(5, 10):  # conv2_1 -> conv3_1
            self.slice2.add_module(str(x), vgg_features[x])
        for x in range(10, 19):  # conv3_1 -> conv4_1
            self.slice3.add_module(str(x), vgg_features[x])
            
        # Set to evaluation mode and don't update weights
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, X):
        h1 = self.slice1(X)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        return h1, h2, h3