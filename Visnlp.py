import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class ViTFeatureExtractor(nn.Module):
    def __init__(self, img_dim_h, img_dim_w, patch_size, embed_dim, num_heads, depth, in_channels=1):
        super().__init__()
        self.vit = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=depth,
            batch_first=True
        )
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_dim_h // patch_size) * (img_dim_w // patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = patches + self.pos_embedding
        x = self.vit(x, x)
        return x.mean(dim=1) 


class BertFeatureExtractor(nn.Module):
    def __init__(self, tokenizer='bert-base-uncased', model='bert-base-uncased', embed_dim=312):
        super().__init__()
        self.model = AutoModel.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.project = nn.Linear(768, embed_dim)

    def forward(self, texts, device='cuda'):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        # dict 내부 tensor만 각각 device로
        inputs = {k: v.to(device) for k, v in inputs.items()}
        self.model.to(device)
        output = self.model(**inputs)
        cls_embedding = output.last_hidden_state[:, 0, :].to(device)
        projected = self.project(cls_embedding)
        return projected


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, query, key, key_padding_mask=None):
        attn_output, _ = self.attention(query, key, key, key_padding_mask=key_padding_mask)
        return attn_output


class VISNLPEXTRACTOR(nn.Module):
    def __init__(self, img_dim_h, img_dim_w, patch_size, embed_dim, num_heads, depth):
        super().__init__()
        self.VITFE = ViTFeatureExtractor(img_dim_h, img_dim_w, patch_size, embed_dim, num_heads, depth, in_channels=3)
        self.BERTFE = BertFeatureExtractor(embed_dim=embed_dim)
        self.cross_attention = CrossAttention(embed_dim, num_heads)

    def forward(self, images, texts, device='cuda'):
        images = images.to(device)
        self.VITFE.to(device)
        self.cross_attention.to(device)

        visual_features = self.VITFE(images)            
        nlp_features = self.BERTFE(texts, device=device)  
        visual_features = visual_features.unsqueeze(1)   
        nlp_features = nlp_features.unsqueeze(1)         
        integrated_features = self.cross_attention(visual_features, nlp_features).squeeze(1)
        return integrated_features
    
class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(max_len, embed_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt, memory):
        """
        tgt: [B, T] - token ids
        memory: [B, D]
        """
        B, T = tgt.size()
        tgt_embed = self.embedding(tgt) + self.pos_embedding[:T, :].unsqueeze(0).to(tgt.device)
        memory = memory.unsqueeze(1)
        output = self.transformer_decoder(tgt_embed, memory)
        return self.fc_out(output) 
    
class CaptionGenerator(nn.Module):
    def __init__(self, visnlp_extractor, vocab_size, embed_dim=512, num_heads=8, ff_dim=2048, num_layers=3):
        super().__init__()
        self.extractor = visnlp_extractor  # VISNLPEXTRACTOR
        self.decoder = CaptionDecoder(vocab_size, embed_dim, num_heads, ff_dim, num_layers)

    def forward(self, images, meta, captions):
        """
        images: [B, 3, H, W]
        meta: list of str (prompt)
        captions: [B, T] token ids
        """
        memory = self.extractor(images, meta) 
        logits = self.decoder(captions, memory)  # [B, T, vocab_size]
        return logits

