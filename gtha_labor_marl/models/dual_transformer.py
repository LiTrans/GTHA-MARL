import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class SpatialTransformer(nn.Module):

    def __init__(
        self,
        agent_dim: int,
        context_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_context_len: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.agent_proj = nn.Linear(agent_dim, d_model)
        self.context_proj = nn.Linear(context_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_context_len, dropout)
        self.cross_attn_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "cross_attn": MultiHeadAttention(d_model, n_heads, dropout),
                        "norm": nn.LayerNorm(d_model),
                        "ff": nn.Sequential(
                            nn.Linear(d_model, d_model * 4),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_model * 4, d_model),
                        ),
                        "ff_norm": nn.LayerNorm(d_model),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        agent_features: torch.Tensor,
        context_features: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        squeeze_output = False
        if agent_features.dim() == 2:
            agent_features = agent_features.unsqueeze(1)
            squeeze_output = True
        batch_size, n_agents, _ = agent_features.shape
        agent_emb = self.agent_proj(agent_features)
        context_emb = self.context_proj(context_features)
        context_emb = self.pos_enc(context_emb)
        x = agent_emb
        for layer in self.cross_attn_layers:
            attn_out = layer["cross_attn"](x, context_emb, context_emb, context_mask)
            x = layer["norm"](x + self.dropout(attn_out))
            ff_out = layer["ff"](x)
            x = layer["ff_norm"](x + self.dropout(ff_out))
        if squeeze_output:
            x = x.squeeze(1)
        return x


class TemporalTransformer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_history_len: int = 36,
        compressed_len: int = 18,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_history_len = max_history_len
        self.compressed_len = compressed_len
        self.input_proj = nn.Linear(input_dim, d_model)
        self.compress_factor = max_history_len // compressed_len
        self.pos_enc = PositionalEncoding(d_model, compressed_len, dropout)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(d_model, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(
        self, history: torch.Tensor, history_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = history.shape
        x = self.input_proj(history)
        if seq_len > self.compressed_len:
            n_chunks = self.compressed_len
            chunk_size = seq_len // n_chunks
            pad_len = n_chunks * chunk_size - seq_len
            if pad_len > 0:
                x = F.pad(x, (0, 0, 0, pad_len))
            x = x[:, : n_chunks * chunk_size, :].view(batch_size, n_chunks, chunk_size, -1)
            x = x.mean(dim=2)
        x = self.pos_enc(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        for layer in self.layers:
            x = layer(x)
        cls_output = x[:, 0, :]
        return self.output_proj(cls_output)


class DualTransformerEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.spatial_transformer = SpatialTransformer(
            agent_dim=config.spatial_input_dim,
            context_dim=config.spatial_input_dim,
            d_model=config.transformer_d_model,
            n_heads=config.transformer_n_heads,
            n_layers=config.transformer_n_layers,
            dropout=config.transformer_dropout,
            max_context_len=config.transformer_max_seq_len,
        )
        self.temporal_transformer = TemporalTransformer(
            input_dim=getattr(config, "history_dim", 5),
            d_model=config.transformer_d_model,
            n_heads=config.transformer_n_heads,
            n_layers=config.transformer_n_layers,
            dropout=config.transformer_dropout,
            max_history_len=getattr(config, "history_length", 36),
            compressed_len=getattr(config, "compressed_history_tokens", 18),
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.transformer_d_model * 2, config.transformer_d_model),
            nn.ReLU(),
            nn.Linear(config.transformer_d_model, config.encoder_output_dim),
        )

    def forward(
        self,
        agent_features: torch.Tensor,
        context_features: torch.Tensor,
        history: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        spatial_out = self.spatial_transformer(agent_features, context_features, context_mask)
        if history is not None and history.size(1) > 0:
            temporal_out = self.temporal_transformer(history)
        else:
            temporal_out = torch.zeros_like(spatial_out)
        fused = torch.cat([spatial_out, temporal_out], dim=-1)
        output = self.fusion(fused)
        return output


class WorkerEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.worker_state_dim, config.encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.encoder_hidden_dim, config.spatial_input_dim),
        )
        self.vacancy_encoder = nn.Sequential(
            nn.Linear(config.vacancy_feature_dim, config.spatial_input_dim), nn.ReLU()
        )
        self.dual_transformer = DualTransformerEncoder(config)
        self.output_dim = config.encoder_output_dim

    def forward(
        self,
        worker_obs: torch.Tensor,
        vacancy_features: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        agent_features = self.feature_encoder(worker_obs)
        if vacancy_features is not None and vacancy_features.size(1) > 0:
            context_features = self.vacancy_encoder(vacancy_features)
        else:
            context_features = torch.zeros(
                worker_obs.size(0), 1, agent_features.size(-1), device=worker_obs.device
            )
        return self.dual_transformer(agent_features, context_features, history)


class FirmEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.firm_state_dim, config.encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.encoder_hidden_dim, config.spatial_input_dim),
        )
        self.applicant_encoder = nn.Sequential(
            nn.Linear(config.applicant_feature_dim, config.spatial_input_dim), nn.ReLU()
        )
        self.dual_transformer = DualTransformerEncoder(config)
        self.output_dim = config.encoder_output_dim

    def forward(
        self,
        firm_obs: torch.Tensor,
        applicant_features: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        agent_features = self.feature_encoder(firm_obs)
        if applicant_features is not None and applicant_features.size(1) > 0:
            context_features = self.applicant_encoder(applicant_features)
        else:
            context_features = torch.zeros(
                firm_obs.size(0), 1, agent_features.size(-1), device=firm_obs.device
            )
        return self.dual_transformer(agent_features, context_features, history)
