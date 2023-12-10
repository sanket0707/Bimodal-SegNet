class CDCA(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super(CDCA, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.query_dim = feature_dim // num_heads
        self.scale = self.query_dim ** -0.5

        # Define linear transformations
        self.W_Q_RGB = nn.Linear(feature_dim, feature_dim)
        self.W_K_RGB = nn.Linear(feature_dim, feature_dim)
        self.W_V_RGB = nn.Linear(feature_dim, feature_dim)

        self.W_Q_event = nn.Linear(feature_dim, feature_dim)
        self.W_K_event = nn.Linear(feature_dim, feature_dim)
        self.W_V_event = nn.Linear(feature_dim, feature_dim)

        # Softmax for attention
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, F_RGB, F_event):
        # Flatten the input feature maps and apply linear embedding
        N, C, H, W = F_RGB.size()
        F_RGB_flat = F_RGB.view(N, C, H * W).transpose(1, 2)  # Flatten and transpose to get N x (H*W) x C
        F_event_flat = F_event.view(N, C, H * W).transpose(1, 2)  # Flatten and transpose

        # Apply linear transformations to get queries, keys, and values
        Q_RGB = self.W_Q_RGB(F_RGB_flat)
        K_RGB = self.W_K_RGB(F_RGB_flat)
        V_RGB = self.W_V_RGB(F_RGB_flat)

        Q_event = self.W_Q_event(F_event_flat)
        K_event = self.W_K_event(F_event_flat)
        V_event = self.W_V_event(F_event_flat)

        # Calculate global context vectors
        G_RGB = torch.matmul(K_RGB.transpose(-2, -1), V_RGB)
        G_event = torch.matmul(K_event.transpose(-2, -1), V_event)

        # Perform cross-attention
        U_RGB = torch.matmul(Q_RGB, self.softmax(G_event * self.scale))
        U_event = torch.matmul(Q_event, self.softmax(G_RGB * self.scale))

        # Reshape attended results to match feature map dimensions and concatenate with residual vectors
        U_RGB_reshaped = U_RGB.transpose(1, 2).view(N, C, H, W)
        U_event_reshaped = U_event.transpose(1, 2).view(N, C, H, W)

        O_RGB = torch.cat((U_RGB_reshaped, F_RGB), dim=1)
        O_event = torch.cat((U_event_reshaped, F_event), dim=1)

        return O_RGB, O_event
