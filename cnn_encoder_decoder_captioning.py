"""
Image Captioning Model: CNN (EfficientNet B1) + Transformer Encoder + GPT-style Decoder
Uses pre-trained EfficientNet B1 as backbone, then transformer encoder on CNN features,
and a GPT-style decoder with cross-attention for caption generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

# EfficientNet B1 backbone (pretrained)
try:
    from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
except ImportError:
    from torchvision.models import efficientnet_b1
    EfficientNet_B1_Weights = None


def _load_efficientnet_b1(pretrained):
    """Load EfficientNet B1, supporting both new (weights=) and old (pretrained=) API."""
    if EfficientNet_B1_Weights is not None and pretrained:
        return efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
    try:
        return efficientnet_b1(weights=None)
    except TypeError:
        return efficientnet_b1(pretrained=False)


class CNNBackbone(nn.Module):
    """
    Pre-trained EfficientNet B1 as CNN backbone.
    Removes the classifier head and returns spatial feature map.
    For 224x224 input -> (B, 1280, 7, 7)
    """
    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()
        backbone = _load_efficientnet_b1(pretrained)
        self.features = backbone.features
        self.out_channels = 1280  # EfficientNet B1 last conv output channels
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x: (B, 3, H, W) -> (B, 1280, H', W')
        return self.features(x)


class TransformerEncoderBlock(nn.Module):
    """Standard transformer encoder block (self-attention + FFN)."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self._attn_block(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def _attn_block(self, x):
        attn_out, _ = self.attn(x, x, x)
        return attn_out


class ImageEncoder(nn.Module):
    """
    CNN backbone + projection + optional transformer encoder.
    Maps image to sequence of embeddings for the decoder to attend to.
    """
    def __init__(
        self,
        cnn_backbone,
        backbone_out_channels=1280,
        embed_dim=512,
        encoder_depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.backbone = cnn_backbone
        # Project CNN feature map to embed_dim and flatten spatial dims
        # Backbone gives (B, 1280, 7, 7) -> we want (B, 49, embed_dim)
        self.projection = nn.Conv2d(backbone_out_channels, embed_dim, kernel_size=1)
        self.n_patches = 7 * 7  # For 224x224 input, EfficientNet B1 gives 7x7
        self.embed_dim = embed_dim

        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(encoder_depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        features = self.backbone(x)  # (B, 1280, 7, 7)
        # Project and flatten: (B, embed_dim, 7, 7) -> (B, embed_dim, 49) -> (B, 49, embed_dim)
        features = self.projection(features)
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)  # (B, 49, embed_dim)
        features = features + self.pos_embed
        features = self.dropout(features)

        for block in self.encoder_blocks:
            features = block(features)
        features = self.norm(features)
        return features


class GPT2DecoderBlock(nn.Module):
    """Decoder block: causal self-attention + cross-attention to encoder + FFN."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, encoder_output, causal_mask=None):
        # Causal self-attention
        self_attn_out, _ = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            attn_mask=causal_mask
        )
        x = x + self_attn_out
        # Cross-attention to encoder (image) features
        cross_out, _ = self.cross_attn(
            self.norm_cross(x), encoder_output, encoder_output
        )
        x = x + cross_out
        x = x + self.mlp(self.norm2(x))
        return x


class GPT2Decoder(nn.Module):
    """GPT-style decoder: token + position embedding, decoder blocks, LM head."""
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            GPT2DecoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, input_ids, encoder_output):
        B, T = input_ids.shape
        token_emb = self.token_embed(input_ids)
        pos_emb = self.pos_embed[:, :T, :]
        x = self.dropout(token_emb + pos_emb)

        # Causal mask: (T, T) lower triangular
        causal_mask = torch.triu(
            torch.ones(T, T, device=input_ids.device) * float('-inf'),
            diagonal=1
        )
        for block in self.blocks:
            x = block(x, encoder_output, causal_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


class CNNEncoderDecoderCaptioning(nn.Module):
    """
    Image captioning: EfficientNet B1 (CNN) + Transformer Encoder + GPT-style Decoder.
    """
    def __init__(
        self,
        vocab_size=50257,
        max_seq_len=77,
        embed_dim=512,
        encoder_depth=6,
        decoder_depth=6,
        num_heads=8,
        dropout=0.1,
        pretrained_cnn=True,
        freeze_cnn_backbone=False,
    ):
        super().__init__()
        cnn = CNNBackbone(pretrained=pretrained_cnn, freeze_backbone=freeze_cnn_backbone)
        self.encoder = ImageEncoder(
            cnn_backbone=cnn,
            backbone_out_channels=cnn.out_channels,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.decoder = GPT2Decoder(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, images, input_ids):
        """
        Args:
            images: (B, 3, H, W)
            input_ids: (B, T) token ids
        Returns:
            logits: (B, T, vocab_size)
        """
        encoder_output = self.encoder(images)
        logits = self.decoder(input_ids, encoder_output)
        return logits

    @torch.no_grad()
    def generate(
        self,
        images,
        start_token_id,
        end_token_id,
        max_length=50,
        temperature=1.0,
        top_k=50,
    ):
        B = images.shape[0]
        device = images.device
        encoder_output = self.encoder(images)
        generated = torch.full((B, 1), start_token_id, dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            logits = self.decoder(generated, encoder_output)
            next_token_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, -1, None]] = float('-inf')
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == end_token_id).all():
                break
        return generated

    @torch.no_grad()
    def generate_beam(
        self,
        images,
        start_token_id,
        end_token_id,
        max_length=50,
        beam_width=5,
    ):
        """
        Generate captions using beam search. Processes one image at a time.
        Returns (B, seq_len) tensor of token ids (best beam per image).
        """
        B = images.size(0)
        device = images.device
        encoder_output = self.encoder(images)  # (B, S, D)
        all_seqs = []
        for b in range(B):
            enc_b = encoder_output[b : b + 1]  # (1, S, D)
            beams = torch.full(
                (beam_width, 1), start_token_id, dtype=torch.long, device=device
            )
            scores = torch.zeros(beam_width, device=device)
            enc_b_expanded = enc_b.expand(beam_width, -1, -1)  # (beam_width, S, D)
            for _ in range(max_length - 1):
                logits = self.decoder(beams, enc_b_expanded)  # (beam_width, L, V)
                next_logits = logits[:, -1, :]  # (beam_width, V)
                log_probs = F.log_softmax(next_logits, dim=-1)
                cand_scores = scores.unsqueeze(1) + log_probs  # (beam_width, V)
                flat_scores = cand_scores.view(-1)
                topk_scores, topk_idx = torch.topk(flat_scores, beam_width)
                scores = topk_scores
                V = log_probs.size(-1)
                beam_idx = topk_idx // V
                token_id = topk_idx % V
                next_tokens = token_id.unsqueeze(1)
                beams = torch.cat([beams[beam_idx], next_tokens], dim=1)
            best_beam_idx = scores.argmax(dim=0).item()
            all_seqs.append(beams[best_beam_idx])
        return torch.stack(all_seqs)
