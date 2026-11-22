import torch
from torch import nn


class BearingCNN(nn.Module):
    def __init__(self, num_tokens=128):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),

            # Block 4
            nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Temporal token compression
        self.token_pool = nn.AdaptiveAvgPool1d(num_tokens)

    def forward(self, windows):
        # Input: (B, 1, 16000)

        x = self.features(windows)
        # (B, 128, 250)

        x = self.token_pool(x)
        # (B, 128, num_tokens)

        embedding = x.transpose(1, 2)
        # (B, num_tokens, 128)

        return embedding


class OperatingConditionMLP(nn.Module):
    def __init__(self, pooled_size=16, num_classes=32):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(pooled_size)

        self.mlp = nn.Sequential(
            nn.Linear(4, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
        )

        self.logits = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)

        )

    def forward(self, medium_signals, temperature, return_logits=False):
        # medium_signals: (B, W, 3, 1000)
        # temperature:    (B, W, 1)

        B, W, C, L = medium_signals.shape

        # Combine batch and window dimensions
        x = medium_signals.reshape(B * W, C, L)
        # (B*W, 3, 1000)

        # Reduce temporal resolution
        x = self.pool(x)
        # (B*W, 3, 16)

        # Convert to temporal tokens
        x = x.transpose(1, 2)
        # (B*W, 16, 3)

        # Broadcast temperature across temporal positions
        temp = temperature.reshape(B * W, 1, 1)
        temp = temp.expand(-1, x.size(1), -1)
        # (B*W, 16, 1)

        # Concatenate operating features
        x = torch.cat([x, temp], dim=-1)
        # (B*W, 16, 4)

        # Encode each temporal position
        embedding = self.mlp(x)
        # (B*W, 16, 128)

        embedding = embedding.reshape(B, W, -1, 128)
        # (B, W, 16, 128)

        if return_logits:
            logits = self.logits(embedding)
            # (B, W, 16, num_classes)

            return embedding, logits

        return embedding


class TemporalAttention(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        max_windows=64
    ):
        super().__init__()

        # Learnable recording-level classification token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02
        )

        # Learnable temporal positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_windows + 1, embed_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Optional classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, embeddings, return_logits=False):
        # embeddings: (B, W, 128)

        B, W, D = embeddings.shape

        if W + 1 > self.pos_embedding.size(1):
            raise ValueError("Number of windows exceeds max_windows")

        # Add recording-level CLS token
        cls = self.cls_token.expand(B, -1, -1)

        x = torch.cat([cls, embeddings], dim=1)
        # (B, W+1, 128)

        # Add temporal positional information
        x = x + self.pos_embedding[:, :W + 1, :]

        # Temporal self-attention
        x = self.temporal_encoder(x)
        # (B, W+1, 128)

        x = self.norm(x)

        # Recording-level embedding
        recording_embedding = x[:, 0, :]
        # (B, 128)

        if return_logits:
            logits = self.classifier(recording_embedding)
            # (B, num_classes)

            return recording_embedding, logits

        return recording_embedding