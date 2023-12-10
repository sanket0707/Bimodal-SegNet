import torch
import torch.nn as nn
import torch.nn.functional as F

class APFABlock(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, weights):
        super(APFABlock, self).__init__()
        self.weights = weights  # weights for each resolution [w1, w2, w3, w4]
        self.branches = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                       dilation=rate, padding=rate) for rate in atrous_rates]
        )
        self.conv_1x1 = nn.Conv2d(len(atrous_rates) * out_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2, x3, x4):
        # Weight and combine feature maps from different stages
        x1_weighted = self.weights[0] * x1
        x2_weighted = self.weights[1] * x2
        x3_weighted = self.weights[2] * x3
        x4_weighted = self.weights[3] * x4
        x_combined = torch.cat((x1_weighted, x2_weighted, x3_weighted, x4_weighted), dim=1)
        
        # Apply dilated convolutions for each branch
        branch_outputs = [branch(x_combined) for branch in self.branches]
        
        # Concatenate the outputs along the channel dimension
        x_new_combined = torch.cat(branch_outputs, dim=1)
        
        # Pass the combined map through a 1x1 convolution
        x_out = self.conv_1x1(x_new_combined)
        
        return x_out


# Define the in_channels, out_channels, atrous_rates, and weights for each resolution
apfa_module = APFABlock(in_channels, out_channels, atrous_rates, weights)
# Pass the weighted and concatenated feature maps from CDCA modules
 x_out = apfa_module(x1, x2, x3, x4)
