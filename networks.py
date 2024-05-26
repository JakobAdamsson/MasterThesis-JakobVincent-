import torch
import torch.nn as nn
from torch.nn import Conv2d,Sequential, ReLU, Linear, MaxPool2d, BatchNorm2d, ConvTranspose2d,Upsample, Sigmoid

def dilation_block(in_channels,out_channels):
    block = Sequential(
    Conv2d(in_channels,out_channels,kernel_size=(3,3),stride=1,padding=1,dilation=1,bias=False), # A dilation of 1 requires only padding = 1 with kernelsize 3 to maintain shape, Norm contains bias
    BatchNorm2d(out_channels),
    ReLU(inplace=True),
    Conv2d(out_channels,out_channels,kernel_size=(3,3),stride=1,padding=1,dilation=1,bias=False),
    BatchNorm2d(out_channels),
    ReLU(inplace=True),
    Conv2d(out_channels,out_channels,kernel_size=(3,3),stride=1,padding=2,dilation=2,bias=False), # A dilation of 2 requires only padding = 2 with kernelsize 3 to maintain shape
    BatchNorm2d(out_channels),
    ReLU(inplace=True),
    Conv2d(out_channels,out_channels,kernel_size=(3,3),stride=1,padding=2,dilation=2,bias=False),
    BatchNorm2d(out_channels),
    ReLU(inplace=True)
    )
    return block

def up_conv(in_channels, out_channels): # This is used to avoid checkerboard artefacts
    return Sequential(
        Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        BatchNorm2d(out_channels),
        ReLU(inplace=True)
    )

class L_U_Net(nn.Module):
    def __init__(self,num_classes,num_filters) -> None:
        super(L_U_Net,self).__init__()
        self.initial_layer = dilation_block(3,num_filters)
        self.dil_block = dilation_block(num_filters,num_filters)
        self.pooling = MaxPool2d(kernel_size=(2,2),stride=(2,2))

        self.bottleneck = Sequential(
            Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1,bias=False),
            BatchNorm2d(num_filters),
            ReLU(inplace=True),

            Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1,bias=False),
            BatchNorm2d(num_filters),
            ReLU(inplace=True),
        )

        self.upsample = up_conv(num_filters,num_filters)
        self.expand_conv = Sequential(
            Conv2d(num_filters*2, num_filters, kernel_size=3, stride=1, padding=1,bias=False), # numfilters*2 due to skip connections :)
            BatchNorm2d(num_filters),
            ReLU(inplace=True),

        )
        self.final_layer = Conv2d(num_filters,num_classes, kernel_size=1)

    def forward(self,x):
        dil1 = self.initial_layer(x)
        pool1 = self.pooling(dil1)

        dil2 = self.dil_block(pool1)
        pool2 = self.pooling(dil2)

        dil3 = self.dil_block(pool2)
        pool3 = self.pooling(dil3)

        dil4 = self.dil_block(pool3)
        pool4 = self.pooling(dil4)

        bottle_neck = self.bottleneck(pool4)

        tconv1 = self.upsample(bottle_neck)
        skip1 = torch.cat((tconv1,dil4),dim=1)
        conv1= self.expand_conv(skip1)

        tconv2= self.upsample(conv1)
        skip2 = torch.cat((tconv2,dil3),dim=1)
        conv2 = self.expand_conv(skip2)

        tconv3 = self.upsample(conv2)
        skip3 = torch.cat((tconv3,dil2),dim=1)
        conv3 = self.expand_conv(skip3)

        tconv4 = self.upsample(conv3)
        skip4 = torch.cat((tconv4,dil1),dim=1)
        conv4 = self.expand_conv(skip4)

        return {'out':self.final_layer(conv4)}

def conv_block(in_channels, out_channels):
    block = nn.Sequential(
        Conv2d(in_channels, out_channels, kernel_size=3, padding=1,bias=False),
        BatchNorm2d(out_channels),
        ReLU(inplace=True),
        Conv2d(out_channels, out_channels, kernel_size=3, padding=1,bias=False),
        BatchNorm2d(out_channels),
        ReLU(inplace=True)
    )
    return block

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.maxpool = MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = conv_block(3, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.bottleneck = conv_block(512, 1024)

        self.upconv1 = up_conv(1024, 512)
        self.decoder1 = conv_block(1024, 512)

        self.upconv2 = up_conv(512, 256)
        self.decoder2 = conv_block(512, 256)

        self.upconv3 = up_conv(256, 128)
        self.decoder3 = conv_block(256, 128)

        self.upconv4 = up_conv(128, 64)
        self.decoder4 = conv_block(128, 64)

        self.final_layer = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Contracting Path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.maxpool(enc1))
        enc3 = self.encoder3(self.maxpool(enc2))
        enc4 = self.encoder4(self.maxpool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.maxpool(enc4))

        # Expanding Path
        up1 = self.upconv1(bottleneck)
        skip1 = torch.cat((up1, enc4), dim=1)
        dec1 = self.decoder1(skip1)

        up2 = self.upconv2(dec1)
        skip2 = torch.cat((up2, enc3), dim=1)
        dec2 = self.decoder2(skip2)

        up3 = self.upconv3(dec2)
        skip3 = torch.cat((up3, enc2), dim=1)
        dec3 = self.decoder3(skip3)

        up4 = self.upconv4(dec3)
        skip4 = torch.cat((up4, enc1), dim=1)
        dec4 = self.decoder4(skip4)

        return {'out':self.final_layer(dec4)}

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = Sequential(
            Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            BatchNorm2d(F_int)
            )

        self.W_x = Sequential(
            Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            BatchNorm2d(F_int)
        )

        self.psi = Sequential(
            Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            BatchNorm2d(1),
            Sigmoid()
        )

        self.relu = ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class Attention_U_Net(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(Attention_U_Net, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)
        self.encoder5 = conv_block(512, 1024)

        self.up_conv5 = up_conv(1024, 512)
        self.att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.decoder5 = conv_block(1024, 512)

        self.up_conv4 = up_conv(512, 256)
        self.att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.decoder4 = conv_block(512, 256)

        self.up_conv3 = up_conv(256, 128)
        self.att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.decoder3 = conv_block(256, 128)

        self.up_conv2 = up_conv(128, 64)
        self.att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.decoder2 = conv_block(128, 64)

        self.final_layer = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        pool1 = self.maxpool(enc1)

        enc2 = self.encoder2(pool1)
        pool2 = self.maxpool(enc2)

        enc3 = self.encoder3(pool2)
        pool3 = self.maxpool(enc3)

        enc4 = self.encoder4(pool3)
        pool4 = self.maxpool(enc4)

        enc5 = self.encoder5(pool4)

        # Decoder I realised this naming convention is easier far too late, changing the previous ones would be painful
        up5 = self.up_conv5(enc5)
        att5 = self.att5(g=up5, x=enc4)
        dec5 = self.decoder5(torch.cat((att5, up5), dim=1))

        up4 = self.up_conv4(dec5)
        att4 = self.att4(g=up4, x=enc3)
        dec4 = self.decoder4(torch.cat((att4, up4), dim=1))

        up3 = self.up_conv3(dec4)
        att3 = self.att3(g=up3, x=enc2)
        dec3 = self.decoder3(torch.cat((att3, up3), dim=1))

        up2 = self.up_conv2(dec3)
        att2 = self.att2(g=up2, x=enc1)
        dec2 = self.decoder2(torch.cat((att2, up2), dim=1))

        output = self.final_layer(dec2)

        return output

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Example usage
    print("L-U-Net")
    input_tensor = torch.randn(1, 3, 224, 224)  # (batch_size, channels, height, width)
    model = L_U_Net(num_classes=4, num_filters=16)
    output_tensor = model(input_tensor)
    print(output_tensor['out'].size(),'\n')  # Output dimensions should match the input spatial dimensions but with num_classes channels

    print("U-Net")
    input_tensor = torch.randn(1, 3, 224, 224)
    model = UNet(num_classes=4)
    output_tensor = model(input_tensor)
    print(output_tensor['out'].size(),'\n')

    print("Attention-U-Net")
    in_channels = 3
    model = Attention_U_Net(4, in_channels)
    input_image = torch.randn(1, 3, 224, 224)
    output = model(input_image)

    print(output.shape)  # Expected output shape: (1, num_classes, 224, 224)
