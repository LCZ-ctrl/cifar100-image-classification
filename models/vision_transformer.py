import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Convert image to patch embedding + class token + positional embedding
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        # calculate number of patches
        self.num_patches = (img_size // patch_size) ** 2

        # patch embedding: linear layer
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # class token: (1, 1, D) -> expand to (B, 1, D) in forward
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # positional embedding: (1, N+1, D) -> learnable
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

    def forward(self, x):
        # Input: [B, 3, 32, 32]
        B = x.shape[0]

        # Step 1: patch embedding
        x = self.patch_embed(x)  # [B, 256, 8, 8]

        # Step 2: flatten patches
        x = x.flatten(2)  # [B, 256, 64]

        # Step 3: transpose
        x = x.transpose(1, 2)  # [B, 64, 256]

        # Step 4: add cls token
        class_token = self.class_token.expand(B, -1, -1)  # [B, 1, 256]
        x = torch.cat([class_token, x], dim=1)  # [B, 65, 256]

        # Step 5: add pos embedding
        x += self.pos_embed  # [B, 65, 256]

        return x


class MultiHeadSelfAttention(nn.Module):
    """
    MHSA module
    Input: [B, N, D]
    Output: [B, N, D]
    """

    def __init__(self, embed_dim=256, num_heads=8, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # dim per head

        # linear layers for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # output linear layer
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape  # [B, 65, 256]

        # Step 1: compute Q, K, V
        q = self.q_proj(x)  # [B, 65, 256]
        k = self.k_proj(x)  # [B, 65, 256]
        v = self.v_proj(x)  # [B, 65, 256]

        # Step 2: split into multi-heads
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, 8, 65, 32]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, 8, 65, 32]
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, 8, 65, 32]

        # Step 3: compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))  # [B, 8, 65, 65]
        scores = scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  # scale

        # Step 4: softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)  # [B, 8, 65, 65]
        attn_weights = self.dropout(attn_weights)

        # Step 5: compute weighted sum of V
        attn_output = torch.matmul(attn_weights, v)  # [B, 8, 65, 32]

        # Step 6: concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, 65, 8, 32]
        attn_output = attn_output.view(B, N, D)  # [B, 65, 256]

        # Step 7: linear projection
        output = self.out_proj(attn_output)  # [B, 65, 256]
        output = self.dropout(output)

        return output


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for Transformer encoder
    Input: (B, N, D)
    Output: (B, N, D)
    """

    def __init__(self, embed_dim=256, mlp_dim=1024, dropout=0.3):
        super().__init__()

        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)  # [B, 65, 256] -> [B, 65, 1024]
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [B, 65, 1024] -> [B, 65, 256]
        x = self.dropout(x)
        return x


class TransformerEncoder(nn.Module):
    """
    Single layer of Transformer encoder
    Input: (B, N, D)
    Output: (B, N, D)
    """

    def __init__(self, embed_dim=256, num_heads=8, mlp_dim=1024, dropout=0.3):
        super().__init__()

        # layer normalization before MHSA
        self.ln1 = nn.LayerNorm(embed_dim)

        # MHSA
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        # layer normalization before MLP
        self.ln2 = nn.LayerNorm(embed_dim)

        # MLP
        self.mlp = MLP(embed_dim, mlp_dim, dropout)

    def forward(self, x):
        # residual connection + MHSA
        x = x + self.mhsa(self.ln1(x))  # [B, 65, 256]

        # residual connection + MLP
        x = x + self.mlp(self.ln2(x))  # [B, 65, 256]

        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer model for image classification
    Input: (B, C, H, W)
    Output: (B, num_classes)
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256, num_heads=8, num_layers=6,
                 mlp_dim=1024, num_classes=100, dropout=0.3):
        super().__init__()

        # patch embedding + class token + positional embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Transformer encoder (stack multiple layers)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        # layer normalization for encoder output
        self.ln = nn.LayerNorm(embed_dim)

        # cls head (linear layer)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Step 1: patch embedding + positional encoding
        # Input: [B, 3, 32, 32]
        x = self.patch_embed(x)  # [B, 65, 256]

        # Step 2: pass through Transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)  # [B, 65, 256]

        # Step 3: layer normalization
        x = self.ln(x)  # [B, 65, 256]

        # Step 4: extract cls token feature
        class_token_feature = x[:, 0, :]  # [B, 256]

        # Step 5: classification head
        logits = self.classifier(class_token_feature)  # [B, 100]

        return logits
