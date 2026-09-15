"""
W2V2-AASIST Model Architecture (Official Specification).

References:
- SpeechAntiSpoofingBenchmarks / W2V2-AASIST
- TakHemlata / SSL_Anti-spoofing (Speaker Odyssey 2022)
- clovaai / aasist (ISCA INTERSPEECH 2022)

SSL Front-End: XLS-R 300M (facebook/wav2vec2-xls-r-300m)
Back-End     : AASIST Heterogeneous Graph Attention Network
Input        : Raw Float32 16kHz audio waveform of shape (batch_size, 64600)
Output       : Logits of shape (batch_size, 2)
               index 0 = spoof / synthetic
               index 1 = bona-fide / real
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from transformers import Wav2Vec2Config, Wav2Vec2Model


class Residual_block(nn.Module):
    """RawNet2 / AASIST Residual Conv Block with SELU & BatchNorm."""
    def __init__(self, nb_filts: Tuple[int, int], first: bool = False):
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm2d(nb_filts[0])
        self.conv1 = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(2, 3), padding=(1, 1))
        self.selu1 = nn.SELU()
        self.bn2 = nn.BatchNorm2d(nb_filts[1])
        self.conv2 = nn.Conv2d(nb_filts[1], nb_filts[1], kernel_size=(2, 3), padding=(0, 1))
        self.selu2 = nn.SELU()

        if nb_filts[0] != nb_filts[1]:
            self.conv_downsample = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(1, 3), padding=(0, 1))
        else:
            self.conv_downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if not self.first:
            x = self.bn1(x)
        x = self.conv1(x)
        x = self.selu1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.selu2(x)

        if self.conv_downsample is not None:
            identity = self.conv_downsample(identity)

        return x + identity


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer for AASIST spectral / temporal feature graph nodes."""
    def __init__(self, in_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = nn.Parameter(torch.FloatTensor(out_dim, 1))
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.SELU(inplace=True)

        nn.init.kaiming_uniform_(self.att_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, N, C)
        att_map = torch.matmul(torch.tanh(self.att_proj(x)), self.att_weight)
        att_map = F.softmax(att_map, dim=1)
        x1 = self.proj_with_att(x * att_map)
        x2 = self.proj_without_att(x)
        x_out = x1 + x2
        org_size = x_out.size()
        x_out = self.bn(x_out.view(-1, org_size[-1])).view(org_size)
        return self.act(x_out)


class HtrgGraphAttentionLayer(nn.Module):
    """Heterogeneous Graph Attention Layer for AASIST."""
    def __init__(self, in_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        self.att_weight11 = nn.Parameter(torch.FloatTensor(out_dim, 1))
        self.att_weight22 = nn.Parameter(torch.FloatTensor(out_dim, 1))
        self.att_weight12 = nn.Parameter(torch.FloatTensor(out_dim, 1))
        self.att_weightM = nn.Parameter(torch.FloatTensor(out_dim, 1))
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.SELU(inplace=True)

        nn.init.kaiming_uniform_(self.att_weight11)
        nn.init.kaiming_uniform_(self.att_weight22)
        nn.init.kaiming_uniform_(self.att_weight12)
        nn.init.kaiming_uniform_(self.att_weightM)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, master: torch.Tensor) -> torch.Tensor:
        p1 = self.proj_type1(x1)
        p2 = self.proj_type2(x2)
        node_feat = p1 + p2

        att_m = torch.tanh(self.att_projM(master))
        att_score_m = torch.matmul(att_m, self.att_weightM)
        node_feat = node_feat * torch.sigmoid(att_score_m)

        x_out = self.proj_with_att(node_feat) + self.proj_without_att(node_feat)
        org_size = x_out.size()
        x_out = self.bn(x_out.view(-1, org_size[-1])).view(org_size)
        return self.act(x_out)


class GraphPooling(nn.Module):
    """Graph Attention Node Pooling."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.proj(x), dim=1)
        return torch.sum(x * weights, dim=1)


class AASISTBackEnd(nn.Module):
    """
    AASIST Spectro-Temporal Graph Attention Back-End Network.
    Matches the exact state_dict structure of LA_model.pth.
    """
    def __init__(self):
        super().__init__()
        # Projection from 1024-dim SSL output to 128-dim AASIST space
        self.LL = nn.Linear(1024, 128)
        self.first_bn = nn.BatchNorm2d(1)
        self.first_bn1 = nn.BatchNorm2d(64)

        # RawNet2 Residual Conv Encoder Blocks
        self.encoder = nn.ModuleList([
            nn.ModuleList([Residual_block((1, 32), first=True)]),
            nn.ModuleList([Residual_block((32, 32))]),
            nn.ModuleList([Residual_block((32, 64))]),
            nn.ModuleList([Residual_block((64, 64))]),
            nn.ModuleList([Residual_block((64, 64))]),
            nn.ModuleList([Residual_block((64, 64))]),
        ])

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=1)
        )

        self.pos_S = nn.Parameter(torch.FloatTensor(1, 42, 64))
        self.master1 = nn.Parameter(torch.FloatTensor(1, 1, 64))
        self.master2 = nn.Parameter(torch.FloatTensor(1, 1, 64))
        nn.init.kaiming_uniform_(self.pos_S)
        nn.init.kaiming_uniform_(self.master1)
        nn.init.kaiming_uniform_(self.master2)

        self.GAT_layer_S = GraphAttentionLayer(in_dim=64, out_dim=64)
        self.GAT_layer_T = GraphAttentionLayer(in_dim=64, out_dim=64)

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(in_dim=64, out_dim=32)
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(in_dim=32, out_dim=32)
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(in_dim=64, out_dim=32)
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(in_dim=32, out_dim=32)

        self.pool_S = GraphPooling(64)
        self.pool_T = GraphPooling(64)
        self.pool_hS1 = GraphPooling(32)
        self.pool_hS2 = GraphPooling(32)
        self.pool_hT1 = GraphPooling(32)
        self.pool_hT2 = GraphPooling(32)

        # Output classifier head (2 logits: index 0 = spoof, index 1 = bona-fide)
        self.out_layer = nn.Linear(160, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x from SSL: shape (B, seq_len, 1024)
        x = self.LL(x)  # (B, seq_len, 128)
        x = x.unsqueeze(1)  # (B, 1, seq_len, 128)
        x = self.first_bn(x)

        for block_group in self.encoder:
            for block in block_group:
                x = block(x)

        x = self.first_bn1(x)  # (B, 64, F, T)

        # Spectral node pooling & Temporal node pooling
        x_s = torch.mean(x, dim=3).transpose(1, 2)  # (B, F, 64)
        x_t = torch.mean(x, dim=2).transpose(1, 2)  # (B, T, 64)

        g_s = self.GAT_layer_S(x_s)  # (B, F, 64)
        g_t = self.GAT_layer_T(x_t)  # (B, T, 64)

        B = x.size(0)
        m1 = self.master1.repeat(B, 1, 1)
        m2 = self.master2.repeat(B, 1, 1)

        st11 = self.HtrgGAT_layer_ST11(g_s, g_s, m1)   # (B, F, 32)
        st12 = self.HtrgGAT_layer_ST12(st11, st11, m1[:, :, :32]) # (B, F, 32)
        st21 = self.HtrgGAT_layer_ST21(g_t, g_t, m2)   # (B, T, 32)
        st22 = self.HtrgGAT_layer_ST22(st21, st21, m2[:, :, :32]) # (B, T, 32)

        p1 = self.pool_hS1(st11)  # (B, 32)
        p2 = self.pool_hS2(st12)  # (B, 32)
        p3 = self.pool_hT1(st21)  # (B, 32)
        p4 = self.pool_hT2(st22)  # (B, 32)
        p5 = self.pool_hS1(m1[:, :, :32])  # (B, 32)

        feat = torch.cat([p1, p2, p3, p4, p5], dim=-1)  # (B, 160)
        logits = self.out_layer(feat)  # (B, 2)
        return logits


class SSLWrapper(nn.Module):
    """Wrapper around HuggingFace Wav2Vec2Model for XLS-R 300M SSL representation extraction."""
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.model = Wav2Vec2Model(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out.last_hidden_state


class W2V2AASIST(nn.Module):
    """
    Full W2V2-AASIST Deepfake Voice Detection Architecture.
    
    Front-End: XLS-R 300M (facebook/wav2vec2-xls-r-300m)
    Back-End : AASIST Graph Attention Classifier
    """
    def __init__(self, ssl_model_name: str = "facebook/wav2vec2-xls-r-300m", config: Optional[Wav2Vec2Config] = None):
        super().__init__()
        self.model_name = "W2V2-AASIST"
        self.backbone_name = "XLS-R 300M"
        self.ssl_model_name = ssl_model_name
        self.required_samples = 64600  # Exactly 64,600 samples (~4.04s @ 16kHz)

        if config is None:
            config = Wav2Vec2Config.from_pretrained(ssl_model_name)

        self.ssl_model = SSLWrapper(config)

        # Attach AASIST backend layers directly to self to match state_dict keys
        backend = AASISTBackEnd()
        for k, v in backend.named_children():
            setattr(self, k, v)
        for k, v in backend.named_parameters(recurse=False):
            setattr(self, k, v)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_values: Raw Float32 16kHz audio tensor of shape (batch_size, 64600)
            
        Returns:
            logits: PyTorch tensor of shape (batch_size, 2)
                    logits[:, 0] = spoof / synthetic
                    logits[:, 1] = bona-fide / real
        """
        ssl_feats = self.ssl_model(input_values)
        logits = AASISTBackEnd.forward(self, ssl_feats)
        return logits
