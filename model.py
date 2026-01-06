import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys


def full_attention_conv(qs, ks, vs, output_attn=False):
    """
    Full (soft) attention without explicit graph structure.
    qs, ks, vs: [N, hidden_dim]
    Returns: [N, hidden_dim] (and attention matrix if requested)
    """
    # L2-normalise queries and keys for stable sigmoid attention
    qs_norm = torch.norm(qs, p=2, dim=-1, keepdim=True)
    ks_norm = torch.norm(ks, p=2, dim=-1, keepdim=True)

    # Pair-wise attention scores: [N, L]
    attention_num = torch.sigmoid(torch.einsum("nd,ld->nl", qs, ks))

    # Normalise rows to sum=1 (softmax-like)
    all_ones = torch.ones(ks.shape[0], device=ks.device)          # [L]
    attention_normaliser = torch.einsum("nl,l->n", attention_num, all_ones)  # [N]
    attention_normaliser = attention_normaliser.unsqueeze(1).repeat(1, ks.shape[0]) + 1e-8  # [N, L]

    attention = attention_num / attention_normaliser               # [N, L]
    attn_output = torch.einsum("nl,ld->nd", attention, vs)       # [N, hidden_dim]

    if output_attn:
        return attn_output, attention_num
    return attn_output


class IntraViewDiffusion(nn.Module):
    """
    Intra-view diffusion: every view updates its own node embeddings
    via full attention among its nodes.
    """
    def __init__(self, in_channels, out_channels, view):
        super(IntraViewDiffusion, self).__init__()
        self.keyProjection   = nn.ModuleList()
        self.queryProjection = nn.ModuleList()
        self.valueProjection = nn.ModuleList()

        for _ in range(view):
            self.keyProjection.append(nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels)
            ))
            self.queryProjection.append(nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels)
            ))
            self.valueProjection.append(nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels)
            ))

        self.out_channels = out_channels
        self.views = view

    def reset_parameters(self):
        for module in [self.keyProjection, self.queryProjection, self.valueProjection]:
            for seq in module:
                for layer in seq:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()

    def forward(self, latent_feature):
        # latent_feature: [view, N, hidden_dim]
        intra_view = []
        for v in range(self.views):
            feat = latent_feature[v]                                # [N, hidden_dim]
            Key   = self.keyProjection[v](feat)
            Query = self.queryProjection[v](feat)
            Value = self.valueProjection[v](feat)
            attn_out = full_attention_conv(Query, Key, Value)       # [N, hidden_dim]
            intra_view.append(attn_out)

        return torch.stack(intra_view, dim=0)                       # [view, N, hidden_dim]


class InterViewDiffusion(nn.Module):
    """
    Inter-view diffusion: every view attends to every other view
    to exchange global information.
    """
    def __init__(self, args, in_channels, out_channels):
        super(InterViewDiffusion, self).__init__()
        self.args = args
        self.WQ = nn.Linear(in_channels, out_channels)
        self.WK = nn.Linear(in_channels, out_channels)
        self.WV = nn.Linear(in_channels, out_channels)
        self.bn_q = nn.BatchNorm1d(out_channels)
        self.bn_k = nn.BatchNorm1d(out_channels)
        self.bn_v = nn.BatchNorm1d(out_channels)

    # ------------------------------------------------------------------
    # Doubly-stochastic projection helpers (ensure rows & cols sum to 1)
    # ------------------------------------------------------------------
    def projection_p(self, P):
        """Row-wise projection onto simplex."""
        dim = P.size(0)
        device = P.device
        one = torch.ones(dim, 1, device=device)
        relu = nn.ReLU()
        P1 = relu(P)
        support1 = torch.mm(torch.mm(P1, one) - one, one.t()) / dim
        P2 = P1 - support1
        support2 = torch.mm(one, torch.mm(one.t(), P2) - one.t()) / dim
        P3 = P2 - support2
        return P3

    def softmax_projection(self, x):
        """Average of row-wise & col-wise softmax."""
        return (F.softmax(x, dim=0) + F.softmax(x, dim=1)) * 0.5

    def multi_projection(self, x):
        """Iterative doubly-stochastic projection."""
        proj_x = self.softmax_projection(x)
        for _ in range(10):
            proj_x = self.projection_p(proj_x)
        return proj_x

    # ------------------------------------------------------------------
    def forward(self, latent_feature):
        # latent_feature: [view, N, hidden_dim]
        view_num, N, hidden_dim = latent_feature.shape

        # Linear + BN while preserving shape
        Q = self.bn_q(self.WQ(latent_feature).permute(1, 0, 2).reshape(-1, hidden_dim))
        Q = Q.reshape(N, view_num, hidden_dim).permute(1, 0, 2)   # [view, N, hidden_dim]
        K = self.bn_k(self.WK(latent_feature).permute(1, 0, 2).reshape(-1, hidden_dim))
        K = K.reshape(N, view_num, hidden_dim).permute(1, 0, 2)
        V = self.bn_v(self.WV(latent_feature).permute(1, 0, 2).reshape(-1, hidden_dim))
        V = V.reshape(N, view_num, hidden_dim).permute(1, 0, 2)

        # Fro-norm normalisation
        q_norm = torch.norm(Q, p='fro', dim=(1, 2), keepdim=True) + 1e-8
        k_norm = torch.norm(K, p='fro', dim=(1, 2), keepdim=True) + 1e-8
        v_norm = torch.norm(V, p='fro', dim=(1, 2), keepdim=True) + 1e-8
        Q, K, V = Q / q_norm, K / k_norm, V / v_norm

        # View-to-view attention
        P = torch.sigmoid(torch.einsum('ivd,jvd->ij', Q, K))  # [view, view]
        P = self.multi_projection(P)                          # doubly-stochastic
        final = torch.einsum('ij,ivd->jvd', P, V)             # [view, N, hidden_dim]
        return final


class ViewAttentionFusion(nn.Module):
    """
    Learnable weighting of views via self-attention.
    """
    def __init__(self, hidden_dim, view_num):
        super(ViewAttentionFusion, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.temperature = nn.Parameter(torch.tensor(math.sqrt(hidden_dim)))

    def forward(self, view_features):
        # view_features: [view, N, hidden_dim]
        view_num, N, hidden_dim = view_features.shape

        # Global view-level descriptor
        view_avg = F.adaptive_avg_pool1d(view_features.permute(1, 2, 0), 1).squeeze(-1).mean(0)  # [hidden_dim]

        q = self.query_proj(view_avg).unsqueeze(0)                # [1, hidden_dim]
        k = self.key_proj(view_features.mean(dim=1))              # [view, hidden_dim]
        attn_scores = torch.einsum('hd,vd->v', q, k) / self.temperature  # [view]
        attn_weights = F.softmax(attn_scores, dim=0)              # [view]

        # Weighted sum over views
        attn_weights = attn_weights.unsqueeze(-1).unsqueeze(-1)   # [view, 1, 1]
        fused = (attn_weights * view_features).sum(0)             # [N, hidden_dim]
        return fused


class GL_DiffBN(nn.Module):
    """
    Overall model: intra-view + inter-view diffusion + view fusion + classifier.
    """
    def __init__(self, in_channels_list, hidden_channels, out_channels, view, args):
        super(GL_DiffBN, self).__init__()
        self.view = len(in_channels_list)

        # Initial non-linear projection per view
        self.classifier = nn.ModuleList()
        for v in range(self.view):
            self.classifier.append(nn.Sequential(
                nn.Linear(in_channels_list[v], hidden_channels),
                nn.ReLU(),
                nn.Dropout(args.dropout)
            ))

        # Diffusion blocks
        self.alpha  = args.alpha
        self.layers = args.layers
        self.Intraconvs = nn.ModuleList()
        self.Intercovs  = nn.ModuleList()
        for _ in range(args.layers):
            self.Intraconvs.append(IntraViewDiffusion(hidden_channels, hidden_channels, self.view))
            self.Intercovs.append(InterViewDiffusion(args, hidden_channels, hidden_channels))

        # Fusion & final classifier
        self.view_attn_fusion = ViewAttentionFusion(hidden_channels, self.view)
        self.final_classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = args.dropout
        self.activation = F.relu

    # --------------------------
    def reset_parameters(self):
        for seq in self.classifier:
            for layer in seq:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
        for module in self.Intraconvs + self.Intercovs:
            module.reset_parameters()
        self.view_attn_fusion.query_proj.reset_parameters()
        self.view_attn_fusion.key_proj.reset_parameters()
        self.final_classifier.reset_parameters()

    # --------------------------
    def forward(self, x_list, return_hidden=False):
        # x_list: list of [N, in_channels_v] for each view
        x_latent = []
        for v in range(self.view):
            x_latent.append(self.classifier[v](x_list[v]))  # [N, hidden_channels]
        x_latent = torch.stack(x_latent, dim=0)             # [view, N, hidden_channels]

        # Diffusion layers
        for _ in range(self.layers):
            intra = self.Intraconvs[_](x_latent)
            inter = self.Intercovs[_](x_latent)
            x_latent = (1 - self.alpha) * x_latent + self.alpha * (intra + inter) * 0.5

        # View fusion
        final_hidden = self.view_attn_fusion(x_latent)      # [N, hidden_channels]
        out = self.final_classifier(final_hidden)           # [N, num_classes]

        if return_hidden:
            return final_hidden, out
        return out