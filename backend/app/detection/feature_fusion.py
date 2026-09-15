"""
Feature-Augmented W2V2-AASIST Model Architecture (Experiment C).

Combines:
1. Pre-trained W2V2-AASIST 160-dim Spectro-Temporal Graph Representation
2. Explicit Acoustic / Spectral Feature Vector (LFCC, CQCC, Spectral, Prosodic, Energy)

Feature Fusion Pipeline:
  [Raw Audio (64600 samples)]
          |
     +----+----+
     |         |
     v         v
 [W2V2-AASIST] [Acoustic Feature Extractor]
     | (160)   | (D_feat)
     |         v
     |   [Acoustic Projection Layer] (64)
     |         |
     +----+----+
          |
  [Concatenation] (224 dims)
          |
  [Fusion Classifier Head]
          |
  [Logits: 0=Spoof, 1=Bona-Fide]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from transformers import Wav2Vec2Config
from app.detection.aasist import W2V2AASIST, AASISTBackEnd
from app.features.acoustic_extractor import AcousticFeatureExtractor, DEFAULT_FEATURE_GROUPS


class AcousticOnlyClassifier(nn.Module):
    """
    Acoustic-Feature-Only Classifier Network (Experiment C2).
    Classifies voice anti-spoofing strictly using explicit acoustic/spectral features.
    """

    def __init__(self, in_dim: int = 191, hidden_dim: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.SELU(),
            nn.Linear(64, 2)
        )

    def forward(self, acoustic_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            acoustic_feats: Float32 tensor of shape (batch_size, in_dim)
        Returns:
            logits: Float32 tensor of shape (batch_size, 2)
        """
        return self.net(acoustic_feats)


class W2V2AASISTFeatureFusion(nn.Module):
    """
    Feature-Augmented W2V2-AASIST Architecture (Experiment C3).
    """

    def __init__(
        self,
        base_aasist_model: Optional[W2V2AASIST] = None,
        feature_groups: Optional[Dict[str, bool]] = None,
        acoustic_proj_dim: int = 64,
        sample_rate: int = 16000
    ):
        super().__init__()
        self.model_name = "Feature_Augmented_W2V2_AASIST"
        self.sample_rate = sample_rate

        # 1. Instantiate or attach W2V2-AASIST baseline backbone
        if base_aasist_model is not None:
            self.w2v2_aasist = base_aasist_model
        else:
            self.w2v2_aasist = W2V2AASIST()

        # 2. Instantiate Acoustic Feature Extractor
        self.feature_extractor = AcousticFeatureExtractor(
            sample_rate=sample_rate,
            feature_groups=feature_groups
        )
        self.acoustic_dim = self.feature_extractor.get_dim()

        # 3. Acoustic Projection Network
        self.acoustic_proj = nn.Sequential(
            nn.Linear(self.acoustic_dim, acoustic_proj_dim),
            nn.LayerNorm(acoustic_proj_dim),
            nn.SELU(),
            nn.Dropout(0.2)
        )

        # 4. Fusion Classification Head (160 AASIST graph dims + 64 Acoustic dims = 224 dims)
        fused_dim = 160 + acoustic_proj_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.SELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )

    def extract_aasist_embeddings(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Extracts 160-dim spectro-temporal graph representation from W2V2-AASIST."""
        ssl_feats = self.w2v2_aasist.ssl_model(audio_tensor)

        # Forward pass through AASIST backend up to graph pooling
        x = self.w2v2_aasist.LL(ssl_feats)
        x = x.unsqueeze(1)
        x = self.w2v2_aasist.first_bn(x)

        for block_group in self.w2v2_aasist.encoder:
            for block in block_group:
                x = block(x)

        x = self.w2v2_aasist.first_bn1(x)

        x_s = torch.mean(x, dim=3).transpose(1, 2)
        x_t = torch.mean(x, dim=2).transpose(1, 2)

        g_s = self.w2v2_aasist.GAT_layer_S(x_s)
        g_t = self.w2v2_aasist.GAT_layer_T(x_t)

        B = x.size(0)
        m1 = self.w2v2_aasist.master1.repeat(B, 1, 1)
        m2 = self.w2v2_aasist.master2.repeat(B, 1, 1)

        st11 = self.w2v2_aasist.HtrgGAT_layer_ST11(g_s, g_s, m1)
        st12 = self.w2v2_aasist.HtrgGAT_layer_ST12(st11, st11, m1[:, :, :32])
        st21 = self.w2v2_aasist.HtrgGAT_layer_ST21(g_t, g_t, m2)
        st22 = self.w2v2_aasist.HtrgGAT_layer_ST22(st21, st21, m2[:, :, :32])

        p1 = self.w2v2_aasist.pool_hS1(st11)
        p2 = self.w2v2_aasist.pool_hS2(st12)
        p3 = self.w2v2_aasist.pool_hT1(st21)
        p4 = self.w2v2_aasist.pool_hT2(st22)
        p5 = self.w2v2_aasist.pool_hS1(m1[:, :, :32])

        aasist_feat = torch.cat([p1, p2, p3, p4, p5], dim=-1)  # (B, 160)
        return aasist_feat

    def compute_acoustic_features(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """Extracts batch acoustic feature vectors on CPU/GPU."""
        device = audio_tensor.device
        audio_np = audio_tensor.detach().cpu().numpy()

        batch_feats = []
        for i in range(audio_np.shape[0]):
            feats = self.feature_extractor.extract_features(audio_np[i])
            batch_feats.append(feats)

        batch_feats_np = np.stack(batch_feats, axis=0)
        return torch.from_numpy(batch_feats_np).float().to(device)

    def forward(
        self,
        audio_tensor: torch.Tensor,
        precomputed_acoustic_feats: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward Pass.

        Args:
            audio_tensor: Float32 audio waveform tensor of shape (batch_size, 64600)
            precomputed_acoustic_feats: Optional precomputed acoustic tensor (batch_size, acoustic_dim)

        Returns:
            logits: Float32 tensor of shape (batch_size, 2)
                    logits[:, 0] = spoof / synthetic
                    logits[:, 1] = bona-fide / real
        """
        # 1. Extract AASIST 160-dim embeddings
        aasist_emb = self.extract_aasist_embeddings(audio_tensor)  # (B, 160)

        # 2. Extract or use precomputed acoustic features
        if precomputed_acoustic_feats is not None:
            ac_feats = precomputed_acoustic_feats
        else:
            ac_feats = self.compute_acoustic_features(audio_tensor)  # (B, D_feat)

        # 3. Project acoustic features
        ac_proj = self.acoustic_proj(ac_feats)  # (B, 64)

        # 4. Feature Fusion
        fused = torch.cat([aasist_emb, ac_proj], dim=-1)  # (B, 224)

        # 5. Output Classification Head
        logits = self.fusion_head(fused)  # (B, 2)
        return logits
