# In cdca.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CDCA(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super(CDCA, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.query_dim = feature_dim // num_heads
        self.scale = self.query_dim ** -0.5

        self.W_Q_RGB = nn.Linear(feature_dim, feature_dim)
        self.W_K_RGB = nn.Linear(feature_dim, feature_dim)
        self.W_V_RGB = nn.Linear(feature_dim, feature_dim)

        self.W_Q_event = nn.Linear(feature_dim, feature_dim)
        self.W_K_event = nn.Linear(feature_dim, feature_dim)
        self.W_V_event = nn.Linear(feature_dim, feature_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, F_RGB_inter, F_event_inter):
        Q_RGB = self.W_Q_RGB(F_RGB_inter)
        K_RGB = self.W_K_RGB(F_RGB_inter)
        V_RGB = self.W_V_RGB(F_RGB_inter)

        Q_event = self.W_Q_event(F_event_inter)
        K_event = self.W_K_event(F_event_inter)
        V_event = self.W_V_event(F_event_inter)

        G_RGB = torch.matmul(K_RGB.transpose(-2, -1), V_RGB)
        G_event = torch.matmul(K_event.transpose(-2, -1), V_event)

        U_RGB = torch.matmul(Q_RGB, self.softmax(G_event * self.scale))
        U_event = torch.matmul(Q_event, self.softmax(G_RGB * self.scale))

        O_RGB = torch.cat((U_RGB, F_RGB_inter), dim=-1)
        O_event = torch.cat((U_event, F_event_inter), dim=-1)

        # Apply a second linear embedding if needed and add to the original feature map
        # Assuming the existence of additional layers for this purpose

        return O_RGB, O_event
