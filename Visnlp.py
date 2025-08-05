import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class ViTFeatureExtractor(nn.Module): # Image to Feature Vector (VIT Encoder)
    def __init__(self, img_dim_h, img_dim_w, patch_size, embed_dim, num_heads, depth, in_channels=1):
        super().__init__()
        # Transformer 설정
        self.vit = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=depth,
            batch_first=True
        )
        # Patch embedding: 채널을 1로 설정
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 포지셔널 임베딩
        num_patches = (img_dim_h // patch_size) * (img_dim_w // patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        # x: [batch_size, 1, img_dim_h, img_dim_w]
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        print("patches shape:", patches.shape)  # <-- 추가
        print("pos_embedding shape:", self.pos_embedding.shape)
        x = patches + self.pos_embedding  # Positional embedding 추가
        x = self.vit(x, x)  # [batch_size, num_patches, embed_dim]
        return x.mean(dim=1)  # [batch_size, embed_dim]


class BertFeatureExtractor(nn.Module): # BERT Model Feature Extractor
    def __init__(self,tokenizer='bert-base-uncased',Model='bert-base-uncased'):
        super().__init__()
        self.model = AutoModel.from_pretrained('bert-base-uncased', force_download=True)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', force_download=True)

    def forward(self, texts, device='cpu'):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device)
        output = self.model(**inputs)
        # CLS 토큰 임베딩만 사용
        cls_embedding = output.last_hidden_state[:, 0, :]  # [B, hidden_dim]
        return cls_embedding.unsqueeze(1)  # [B, 1, hidden_dim]


class CrossAttention(nn.Module): # Vision Vector + NLP Vector
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, key_padding_mask=None):
        attn_output, _ = self.attention(query, key, key, key_padding_mask=key_padding_mask)
        return attn_output

class VISNLPEXTRACTOR(nn.Module):
    def __init__(self, img_dim_h,img_dim_w, patch_size, embed_dim, num_heads, depth):
        super().__init__()
        self.VITFE = ViTFeatureExtractor(img_dim_h, img_dim_w, patch_size, embed_dim, num_heads, depth, in_channels=3)
        self.BERTFE = BertFeatureExtractor()
        self.cross_attention = CrossAttention(embed_dim, num_heads)

    def forward(self, images, texts):
        # Feature extraction
        visual_features = self.VITFE(images)  # [batch_size, embed_dim]
        nlp_features = self.BERTFE(texts)  # [batch_size, embed_dim]

        # Cross attention
        visual_features = visual_features.unsqueeze(1)  # [batch_size, 1, embed_dim]
        nlp_features = nlp_features.unsqueeze(1)  # [batch_size, 1, embed_dim]
        integrated_features = self.cross_attention(visual_features, nlp_features).squeeze(1)
        
        return integrated_features