import torch
import torch.nn as nn

## Model
class Unet(nn.Module):
    def __init__(self,initial_channel):
        super().__init__()
        
        self.maxpool = self.maxpool_2x2()
        
        self.encoder_l1 = self.double_conv_3x3(in_channel=1,out_channel = initial_channel)
        self.encoder_l2 = self.double_conv_3x3(in_channel=initial_channel,out_channel=initial_channel*2)
        self.encoder_l3 = self.double_conv_3x3(in_channel=initial_channel*2,out_channel=initial_channel*4)
        self.encoder_l4 = self.double_conv_3x3(in_channel=initial_channel*4,out_channel=initial_channel*8)
        self.encoder_l5 = self.double_conv_3x3(in_channel=initial_channel*8,out_channel=initial_channel*16)
        
        
        self.decoder_l1 = self.double_conv_3x3(in_channel=initial_channel*2,out_channel=initial_channel)
        self.decoder_l2 = self.double_conv_3x3(in_channel=initial_channel*4,out_channel=initial_channel*2)
        self.decoder_l3 = self.double_conv_3x3(in_channel=initial_channel*8,out_channel=initial_channel*4)
        self.decoder_l4 = self.double_conv_3x3(in_channel=initial_channel*16,out_channel=initial_channel*8)
        
        self.decoder_upConv_l1 = self.transpose_conv_2x2(in_channel=initial_channel*2,out_channel=initial_channel)
        self.decoder_upConv_l2 = self.transpose_conv_2x2(in_channel=initial_channel*4,out_channel=initial_channel*2)
        self.decoder_upConv_l3 = self.transpose_conv_2x2(in_channel=initial_channel*8,out_channel=initial_channel*4)
        self.decoder_upConv_l4 = self.transpose_conv_2x2(in_channel=initial_channel*16,out_channel=initial_channel*8)
        
        self.seg = nn.Conv2d(in_channels=initial_channel,out_channels=1,kernel_size=1)
        self.activation = nn.Sigmoid()
                    
        
    def forward(self,x):
        x1 = self.encoder_l1(x)
        x2 = self.encoder_l2(self.maxpool(x1))
        x3 = self.encoder_l3(self.maxpool(x2))
        x4 = self.encoder_l4(self.maxpool(x3))
        x5 = self.encoder_l5(self.maxpool(x4))
        
        x = self.decoder_upConv_l4(x5)
        x = torch.concat((x4,x),dim=1)
        x = self.decoder_l4(x)
        x = self.decoder_upConv_l3(x)
        x = torch.concat((x3,x),dim=1)
        x = self.decoder_l3(x)
        x = self.decoder_upConv_l2(x)
        x = torch.concat((x2,x),dim=1)
        x = self.decoder_l2(x)
        x = self.decoder_upConv_l1(x)
        x = torch.concat((x1,x),dim=1)
        x = self.decoder_l1(x)
        
        x = self.seg(x)
        x = self.activation(x)
        
        return x
               
    
    def double_conv_3x3(self,in_channel,out_channel):
        return nn.Sequential(nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,padding='same',bias=False),
                             nn.BatchNorm2d(num_features=out_channel),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,padding='same',bias=False),
                             nn.BatchNorm2d(num_features=out_channel),
                             nn.ReLU(inplace=True))
        
    def maxpool_2x2(self):
        return nn.MaxPool2d(kernel_size=2,stride=2)
    
    def transpose_conv_2x2(self,in_channel,out_channel):
        return nn.ConvTranspose2d(in_channels=in_channel,out_channels=out_channel,kernel_size=2,stride=2)
    

class Unet_Dropout(Unet):
    def __init__(self,initial_channel):
        super().__init__(initial_channel=initial_channel)
        self.encoder_l5 = self.double_conv_3x3(in_channel=initial_channel*8,out_channel=initial_channel*16,drop=True)
        self.decoder_l3 = self.double_conv_3x3(in_channel=initial_channel*8,out_channel=initial_channel*4,drop=True)
        self.decoder_l4 = self.double_conv_3x3(in_channel=initial_channel*16,out_channel=initial_channel*8,drop=True)
        
        
    def double_conv_3x3(self,in_channel,out_channel,drop = False):
        if not drop:
            return nn.Sequential(nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,padding='same',bias=False),
                                nn.BatchNorm2d(num_features=out_channel),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,padding='same',bias=False),
                                nn.BatchNorm2d(num_features=out_channel),
                                nn.ReLU(inplace=True))
        else:
            return nn.Sequential(nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=3,padding='same',bias=False),
                                nn.BatchNorm2d(num_features=out_channel),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=3,padding='same',bias=False),
                                nn.BatchNorm2d(num_features=out_channel),
                                nn.ReLU(inplace=True),
                                nn.Dropout2d(p=.5))
            
    
    
## Loss Function
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,predict,truth,val_loss = False,smooth = 1):
        predict = predict.view(-1)
        truth = truth.view(-1)
        
        if val_loss:
            dice = (2 *(predict * truth).sum())/(predict.sum() + truth.sum() + smooth)
            dice_val = (2 *((predict > 0.5).float() * truth).sum())/(predict.sum() + truth.sum() + smooth)
            return 1-dice, dice_val
        
        intersection = (predict*truth).sum()
        dice = (2 *intersection)/(predict.sum() + truth.sum() + smooth)
        return 1-dice