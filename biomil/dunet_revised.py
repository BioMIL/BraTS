from torchvision.transforms import CenterCrop
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
import torch
import torch.nn as nn
import torch

class D_SE_Add(nn.Module):
    def __init__(self, avgpool_ksize, se_block_out_Cs, in3d_depth=2, in3d_Cs=64, out_Cs=64):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(in_channels=in3d_Cs, out_channels=1, kernel_size=1, padding="same")
        self.conv2d_1 = nn.Conv2d(in_channels=in3d_depth, out_channels=out_Cs, kernel_size=3, padding="same")
        self.conv2d_2 = nn.Conv2d(in_channels=out_Cs*2, out_channels=out_Cs, kernel_size=3, padding="same")
        self.relu = nn.ReLU()
        self.se_block_1 = Squeeze_Excite_Block(avgpool_ksize, filters=se_block_out_Cs, ratio=16)
        self.se_block_2 = Squeeze_Excite_Block(avgpool_ksize, filters=se_block_out_Cs, ratio=16)

    def forward(self, input3d, input2d):
        # avgpool_ksize = input3d.size()[-1]
        x = self.conv3d_1(input3d)
        x = torch.squeeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.se_block_1(x)
        input2d = self.se_block_2(input2d)
        x = torch.cat((x, input2d), dim=1)
        x = self.conv2d_2(x)
        x = self.relu(x)
        return x

class Squeeze_Excite_Block(nn.Module):
    def __init__(self, avgpool_ksize, filters=64, ratio=16) -> None:
        super().__init__()
        self.avgpool2d =  nn.AvgPool2d(avgpool_ksize)#.cuda()
        self.relu = nn.ReLU()#.cuda()
        self.linear_1 = nn.Linear(filters, filters//ratio, bias=False)#.cuda()
        self.linear_2 = nn.Linear(filters//ratio, filters, bias=False)#.cuda()

    def forward(self, x):   
        bs, filters = x.size()[:2]
        se_shape = (bs, 1, 1, filters)
        se = self.avgpool2d(x)   
        se = torch.reshape(se, se_shape) 
        se = self.linear_1(se)
        se = self.relu(se)
        se = self.linear_2(se)
        se = torch.sigmoid(se)
        se = se.permute(0, 3, 1, 2)
        out = torch.mul(x, se) 
        return out

class Bn_Block(nn.Module):
    def __init__(self, in_filters, out_filters) -> None:
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=3, padding='same')#.cuda()
        self.bn2d_1 = nn.BatchNorm2d(num_features=out_filters, eps=1e-03, momentum=0.99)#.cuda()
        self.relu = nn.ReLU()#.cuda()
        self.conv2d_2 = nn.Conv2d(in_channels=out_filters, out_channels=out_filters, kernel_size=3, padding='same')#.cuda()
        self.bn2d_2 = nn.BatchNorm2d(num_features=out_filters, eps=1e-03, momentum=0.99)#.cuda()

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.bn2d_1(x)
        x1 = self.relu(x)
        x = self.conv2d_2(x1)
        x = self.bn2d_2(x)
        self.relu(x)
        return x

class Bn_Block3d(nn.Module):
    def __init__(self, in_filters, out_filters) -> None:
        super().__init__()
        self.conv3d_1 = nn.Conv3d(in_channels=in_filters, out_channels=out_filters, kernel_size=3, padding='same')#.cuda()
        self.bn3d_1 = nn.BatchNorm3d(num_features=out_filters, eps=1e-03, momentum=0.99)#.cuda()
        self.relu = nn.ReLU()#.cuda()
        self.conv3d_2 = nn.Conv3d(in_channels=out_filters, out_channels=out_filters, kernel_size=3, padding='same')#.cuda()
        self.bn3d_2 = nn.BatchNorm3d(num_features=out_filters, eps=1e-03, momentum=0.99)#.cuda()

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.bn3d_1(x)
        x1 = self.relu(x)
        x = self.conv3d_2(x1)
        x = self.bn3d_2(x)
        x = self.relu(x)
        return x

class D_Unet(nn.Module):
    def __init__(self, multilabels=False) -> None:
        super().__init__()
        print("[INFO] This is the revised version of D-Unet")
        self.multilabels = multilabels
        # encode
        self.bn_block3d_1 = Bn_Block3d(in_filters=1, out_filters=32)
        self.bn_block3d_2 = Bn_Block3d(in_filters=32, out_filters=64)
        self.bn_block3d_3 = Bn_Block3d(in_filters=64, out_filters=128)
        
        self.bn_block2d_1 = Bn_Block(in_filters=4, out_filters=32)
        self.bn_block2d_2 = Bn_Block(in_filters=32, out_filters=64)
        self.bn_block2d_3 = Bn_Block(in_filters=64, out_filters=128)
        self.bn_block2d_4 = Bn_Block(in_filters=128, out_filters=256)
        self.bn_block2d_5 = Bn_Block(in_filters=256, out_filters=512)
        
        self.d_se_add_1 = D_SE_Add(96, 64, 2, 64, 64)# (avgpool_ksize, se_block_out_Cs, in3d_depth=2, in3d_Cs=64, out_Cs=64)
        self.d_se_add_2 = D_SE_Add(48, 128, 1, 128, 128)
        
        # decode
        self.bn_block2d_6 = Bn_Block(in_filters=512, out_filters=256)
        self.bn_block2d_7 = Bn_Block(in_filters=256, out_filters=128)
        self.bn_block2d_8 = Bn_Block(in_filters=128, out_filters=64)
        self.bn_block2d_9 = Bn_Block(in_filters=64, out_filters=32)
        
        self.conv2d_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding="same", padding_mode='zeros')
        self.conv2d_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding="same", padding_mode='zeros')
        self.conv2d_3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding="same", padding_mode='zeros')
        self.conv2d_4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding="same", padding_mode='zeros')
        if self.multilabels:
            self.conv2d_5 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1, padding="same", padding_mode='zeros')
        else:
            self.conv2d_5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding="same", padding_mode='zeros')

        self.maxpool3d = nn.MaxPool3d(kernel_size=2)
        self.maxpool2d = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        # self.upsampling2d = nn.Upsample(scale_factor=2)
        self.upsampling2d_1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2)
        self.upsampling2d_2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.upsampling2d_3 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2)
        self.upsampling2d_4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        # 3D branch
        input3d = torch.unsqueeze(x, 1) # (B, 1, 4, 192, 192)
        conv3d1 = self.bn_block3d_1(input3d)# (B, 32, 4, 192, 192)
        pool3d1 = self.maxpool3d(conv3d1) # (B, 32, 2, 96, 96)
        conv3d2 = self.bn_block3d_2(pool3d1)# (B, 64, 2, 96, 96)
        pool3d2 = self.maxpool3d(conv3d2)
        conv3d3 = self.bn_block3d_3(pool3d2)

        # 2D branch
        conv1 = self.bn_block2d_1(x)
        pool1 = self.maxpool2d(conv1)
        conv2 = self.bn_block2d_2(pool1)

        # D_SE_Add Block
        conv2 = self.d_se_add_1(conv3d2, conv2)
        pool2 = self.maxpool2d(conv2)

        conv3 = self.bn_block2d_3(pool2)
        conv3 = self.d_se_add_2(conv3d3, conv3)
        pool3 = self.maxpool2d(conv3)

        conv4 = self.bn_block2d_4(pool3)
        conv4 = self.dropout(conv4)
        pool4 = self.maxpool2d(conv4)

        conv5 = self.bn_block2d_5(pool4)
        conv5 = self.dropout(conv5) # (B, 512, 12, 12)

        # Decoder
        up6 =  self.conv2d_1(self.upsampling2d_1(conv5)) # (B, 256, 24, 24)
        merge6 = torch.cat((conv4, up6), dim=1) # (B, 512, 24, 24)
        conv6 = self.bn_block2d_6(merge6) # (B, 256, 24, 24)

        up7 = self.conv2d_2(self.upsampling2d_2(conv6)) # (B, 128, 48, 48)
        merge7 = torch.cat((conv3, up7), dim=1) # (B, 256, 48, 48)
        conv7 = self.bn_block2d_7(merge7) # (B, 128, 48, 48)

        up8 = self.conv2d_3(self.upsampling2d_3(conv7)) # (B, 64, 96, 96)
        merge8 = torch.cat((conv2, up8), dim=1) # (B, 128, 96, 96)
        conv8 = self.bn_block2d_8(merge8) # (B, 64, 96, 96)

        up9 = self.conv2d_4(self.upsampling2d_4(conv8)) # (B, 32, 128, 128)
        merge9 = torch.cat((conv1, up9), dim=1) # (B, 64, 128, 128)
        conv9 = self.bn_block2d_9(merge9)

        conv10 = self.conv2d_5(conv9)  
        # out = torch.sigmoid(conv10)

        return conv10

# #%%
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# x = torch.rand(1, 4, 192, 192).to(DEVICE)  
# model = D_Unet().to(DEVICE)
# pred = model(x)
# print(pred.size())
# #%%
# x = torch.rand(1, 4, 192, 192).to(DEVICE)
# x = torch.unsqueeze(x, 1) # (B, 1, 4, 192, 192)
# bn3d = Bn_Block3d(in_filters=1, out_filters=32).to(DEVICE)
# out = bn3d(x)

# %%
