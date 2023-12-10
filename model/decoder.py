class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, middle_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_final = nn.Conv2d(middle_channels + out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = self.conv_relu(x)
        x = torch.cat((x, skip), dim=1)
        x = self.conv_final(x)
        return x

class BimodalSegNetDecoder(nn.Module):
    def __init__(self, num_classes):
        super(BimodalSegNetDecoder, self).__init__()
        # Define decoder blocks with the appropriate channel sizes
        self.decoder_block1 = DecoderBlock(512, 256, 256)
        self.decoder_block2 = DecoderBlock(256, 128, 128)
        self.decoder_block3 = DecoderBlock(128, 64, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x_out, skips):
        # skips is a list of the skip connections from the encoder
        x = x_out  # Starting with the output from the APFA module
        x = self.decoder_block1(x, skips[3])  # First skip connection
        x = self.decoder_block2(x, skips[2])  # Second skip connection
        x = self.decoder_block3(x, skips[1])  # Third skip connection
        # Apply final 1x1 conv to get to the number of classes
        x_seg = self.final_conv(x)
        return x_seg


# Assuming `x_out` is the output from the APFA module
# and `skips` is a list of feature maps from the encoder at various stages
 decoder = BimodalSegNetDecoder(num_classes=number_of_classes)
 x_seg = decoder(x_out, skips)
