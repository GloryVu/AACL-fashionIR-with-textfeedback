from torchvision.models.swin_transformer import SwinTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import math

class MultiLevelSwinTransformer(nn.Module):
    def __init__(self, token_dim, proj_dim):
        """
        Args:
            swin_model: Pretrained Swin Transformer model
            token_dim: The dimension of image tokens output from the Swin Transformer
            proj_dim: The target dimension for the final image representation
        """
        super(MultiLevelSwinTransformer, self).__init__()
        
        # Swin transformer to extract features from images
        self.swin_model = models.swin_b(pretrained=True)
        
        self.stage3_output = None
        self.stage4_output = None

        # Register hooks to capture the outputs from Stage 3 and Stage 4
        self.swin_model.features[-2].register_forward_hook(self._hook_stage3)  # Stage 3 hook
        self.swin_model.features[-1].register_forward_hook(self._hook_stage4)  # Stage 4 hook

        # Linear projection to map concatenated tokens to the desired dimension (proj_dim)
        self.projection = nn.Linear(token_dim, proj_dim)

    def _hook_stage3(self, module, input, output):
        """
        Hook to capture the output of Stage 3.
        """
        self.stage3_output = output

    def _hook_stage4(self, module, input, output):
        """
        Hook to capture the output of Stage 4.
        """
        self.stage4_output = output

    def forward(self, x):
        # Run forward pass through Swin Transformer (hooks will capture intermediate outputs)
        _ = self.swin_model(x)

        # Ensure both stage outputs are captured
        if self.stage3_output is None or self.stage4_output is None:
            raise ValueError("Stage outputs not captured properly.")

        # Concatenate features along the token dimension
        concatenated_features = torch.cat((self.stage3_output.view(-1,49,1024), self.stage4_output.view(-1,49,1024)), dim=1)  # Concatenation along token dim
        # Apply linear projection to the concatenated features
        projected_features = self.projection(concatenated_features)

        return projected_features

class TextEnconder(nn.Module):
    def __init__(self,device):        
        super(TextEnconder, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.device = device
    def forward(self, batched_input):
        inputs = [self.tokenizer(i, padding=False, truncation=False, return_tensors="pt") for i in batched_input]
        # Forward pass through the model
        inputs = [i.to(self.device) for i in inputs]
        outputs = [self.distilbert(**i) for i in inputs]
        
        # The output will contain the last hidden states for all sentences in the batch
        return [output.last_hidden_state.squeeze(0) for output in outputs]
    
    # Additive Self-Attention Layer
class AdditiveSelfAttention(nn.Module):
    def __init__(self, d_model):
        super(AdditiveSelfAttention, self).__init__()
        self.d_model = d_model
        self.w_h = nn.Parameter(torch.randn(d_model))  # Learnable weight vector

        # Linear transformations
        self.Fh = nn.Linear(d_model, d_model)
        self.Fo = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(self, tokens):
        N = tokens.size(1)  # Number of tokens (N)
        
        # Linear transformation of input tokens
        h = self.Fh(tokens)
        
        # Compute attention weights (alpha)
        scores = (h @ self.w_h) * self.scale
        alpha = F.softmax(scores, dim=1)  # Shape (batch_size, N)
        
        # Context vector c as weighted sum of token hidden states
        c = torch.sum(alpha.unsqueeze(-1) * h, dim=1)  # Shape (batch_size, d_model)
        
        # Hadamard product: reuse global context information
        v = c.unsqueeze(1) * h  # Shape (batch_size, N, d_model)

        # Transform using another linear layer
        transformed_v = self.Fo(v)

        # Final output: adding h and transformed_v (additive self-attention)
        output = h + transformed_v
        return output

# Composition Block (Multi-head Additive Attention + FFN + Residuals)
class CompositionBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden_dim, num_layers):
        super(CompositionBlock, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.num_layers = num_layers

        # Multi-head additive self-attention layers
        self.attention_layers = nn.ModuleList([AdditiveSelfAttention(d_model) for _ in range(num_heads)])
        self.linear_projection = nn.Linear(num_heads * d_model, d_model)
        # Feed-forward network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, d_model)
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        # Residual connections
        self.residual_connection = nn.Identity()

    def forward(self, batched_visiolinguistic_embeddings):
        # Apply multi-head additive self-attention
        batched_outputs = []
        for visiolinguistic_rep in batched_visiolinguistic_embeddings:
            attention_outputs = [attn(visiolinguistic_rep.unsqueeze(0)) for attn in self.attention_layers]
            concat_attention = torch.cat(attention_outputs, dim=-1)
            concat_attention = self.linear_projection(concat_attention)
            # Apply residual connection and layer normalization
            residual_output = self.residual_connection(visiolinguistic_rep + concat_attention)
            norm_output = self.layer_norm1(residual_output)
    
            # Apply feed-forward network (FFN)
            ffn_output = self.ffn(norm_output)
    
            # Apply residual connection and layer normalization again
            final_output = self.layer_norm2(ffn_output + norm_output)
            batched_outputs.append(final_output)
        return torch.cat([output[:,:98,:] for output in batched_outputs],dim=0)    
class AACL(nn.Module):
    def __init__(self,device):        
        super(AACL, self).__init__()
        self.image_encoder = MultiLevelSwinTransformer(1024,768)
        self.text_encoder = TextEnconder(device=device)
        self.text_encoder.to(device)
        self.text_encoder.distilbert.to(device)
        self.image_encoder.swin_model.to(device)
        self.composition_block = CompositionBlock(d_model=768, num_heads=8, ff_hidden_dim=768, num_layers=3)
        self.device = device
    def forward(self, batched_images, batched_text):
        image_embeddings = self.image_encoder(batched_images)
        text_embeddings = self.text_encoder(batched_text)
        for text_embedding in text_embeddings:
            text_embedding.to(self.device)
        visiolinguistic_embeddings = [torch.cat((image_embedding,text_embedding), dim=0) for image_embedding,text_embedding in zip(image_embeddings,text_embeddings)]
        # The output will contain the last hidden states for all sentences in the batch
        outputs = self.composition_block(visiolinguistic_embeddings)
        return outputs
    
# aacl = AACL()
# print(aacl(torch.rand(2,3,224,224),["make the neck higher", 'Im joke :D']).shape)
