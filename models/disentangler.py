import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatioTemporalConsistencyDisentangler(nn.Module):
    def __init__(self, seq_len, num_landmarks, feat_dim=2, hidden_dim=128, 
                 consistency_lambda=1.0, independence_lambda=0.1, variance_lambda=5):
        super(SpatioTemporalConsistencyDisentangler, self).__init__()
        self.seq_len = seq_len
        self.num_landmarks = num_landmarks
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.consistency_lambda = consistency_lambda
        self.independence_lambda = independence_lambda
        self.variance_lambda = variance_lambda

        self.spatiotemporal_encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 1)),  
            
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)), 
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

        self.conv_output_size = self._get_conv_output_size(seq_len, num_landmarks, feat_dim)

        self.consistency_encoder = nn.Sequential(
            nn.Linear(self.conv_output_size, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        self.variance_encoder = nn.Sequential(
            nn.Linear(self.conv_output_size, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, num_landmarks * feat_dim),  
        )

    def _get_conv_output_size(self, seq_len, num_landmarks, feat_dim):
        dummy_input = torch.zeros(1, 1, seq_len, num_landmarks, feat_dim)
        output = self.spatiotemporal_encoder(dummy_input)
        return output.size(1) * output.size(3) * output.size(4)

    def reshape_input(self, x):
        return x.unsqueeze(1) 

    def forward(self, x):
        batch_size = x.shape[0]
        real_seq_len = x.shape[1]
        
        x_reshaped = self.reshape_input(x)  
        spatiotemporal_feat = self.spatiotemporal_encoder(x_reshaped)  
        
        B, C, T, H, W = spatiotemporal_feat.size()
        spatiotemporal_feat = spatiotemporal_feat.permute(0, 2, 1, 3, 4)  
        spatiotemporal_feat = spatiotemporal_feat.reshape(B, T, -1)  
        target_feat = self.consistency_encoder(spatiotemporal_feat)  
        driving_feat = self.variance_encoder(spatiotemporal_feat)      

        combined = torch.cat([target_feat, driving_feat], dim=2)  
        reconstructed_flat = self.decoder(combined) 
        if not self.training: reconstructed = reconstructed_flat.view(batch_size, real_seq_len, 
                                               self.num_landmarks, self.feat_dim)
        else: reconstructed = reconstructed_flat.view(batch_size, self.seq_len, 
                                               self.num_landmarks, self.feat_dim)
        
        return target_feat, driving_feat, reconstructed

    def compute_temporal_consistency(self, target_feat, driving_feat):
        def fluctuation_coeff(x, dim):
            std_val = torch.std(x, dim=dim, keepdim=True)
            abs_mean = torch.mean(torch.abs(x), dim=dim, keepdim=True)
            return (std_val / abs_mean).mean()
    
        target_fluc = fluctuation_coeff(target_feat, dim=1)
        driving_fluc = fluctuation_coeff(driving_feat, dim=1)
    
        target_mean = torch.mean(target_feat, dim=1) 
        target_batch_fluc = fluctuation_coeff(target_mean, dim=0)
    
        return target_fluc, driving_fluc, target_batch_fluc

    def get_loss(self, x, reconstructed, target_feat, driving_feat):
        rec_loss = F.mse_loss(reconstructed, x)

        target_fluc, driving_fluc, target_batch_fluc= self.compute_temporal_consistency(target_feat, driving_feat)
        target_loss = 0.5*target_fluc + torch.exp(-5*target_batch_fluc)
        driving_loss = torch.exp(-5*driving_fluc)
      
        B, T, H = target_feat.shape
        target_flat = target_feat.view(-1, H)
        driving_flat = driving_feat.view(-1, H)
        
        target_centered = target_flat - target_flat.mean(0, keepdim=True)
        driving_centered = driving_flat - driving_flat.mean(0, keepdim=True)
        cov = torch.mm(target_centered.T, driving_centered) / (B*T - 1)
        independence_loss = torch.norm(cov, p='fro')
        
        total_loss = (rec_loss + 
                     self.consistency_lambda * target_loss +                     
                     self.variance_lambda * driving_loss +
                     self.independence_lambda * independence_loss)
        return total_loss, rec_loss, target_loss, driving_loss, independence_loss




