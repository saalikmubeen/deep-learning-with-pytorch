import torch
from torch import nn


# 1. Create a class called PatchEmbedding
class PatchEmbedding(nn.Module):
  # 2. Initilaize the layer with appropriate hyperparameters
  def __init__(self,
               in_channels: int = 3,
               patch_size: int = 16,
               embedding_dim: int = 768):  # from Table 1 for ViT-Base
    super().__init__()

    self.patch_size = patch_size

    # 3. Create a layer to turn an image into embedded patches
    self.patcher = nn.Conv2d(in_channels=in_channels,
                             out_channels=embedding_dim,
                             kernel_size=patch_size,
                             stride=patch_size,
                             padding=0)

    # 4. Create a layer to flatten feature map outputs of Conv2d
    self.flatten = nn.Flatten(start_dim=2,
                              end_dim=3)

  # 5. Define a forward method to define the forward computation steps
  def forward(self, x):
    # Create assertion to check that inputs are the correct shape
    image_resolution = x.shape[-1]
    assert image_resolution % self.patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

    # Perform the forward pass
    x_patched = self.patcher(x)
    x_flattened = self.flatten(x_patched)
    # 6. Make the returned sequence embedding dimensions are in the right order (batch_size, number_of_patches, embedding_dimension)
    return x_flattened.permute(0, 2, 1)


class MultiHeadSelfAttentionBlock(nn.Module):
  """Creates a multi-head self-attention block ("MSA block" for short).
  """

  def __init__(self,
               # Hidden size D (embedding dimension) from Table 1 for ViT-Base
               embedding_dim: int = 768,
               num_heads: int = 12,  # Heads from Table 1 for ViT-Base
               attn_dropout: int = 0):
    super().__init__()

    # Create the norm layer (LN)
    self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

    # Create multihead attention (MSA) layer
    self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                num_heads=num_heads,
                                                dropout=attn_dropout,
                                                # is the batch first? (batch, seq, feature) -> (batch, number_of_patches, embedding_dimension)
                                                batch_first=True)

  def forward(self, x):
    x = self.layer_norm(x)
    attn_output, _ = self.multihead_attn(query=x,
                                         key=x,
                                         value=x,
                                         need_weights=False)
    return attn_output


class MLPBlock(nn.Module):
  def __init__(self,
               embedding_dim: int = 768,
               mlp_size: int = 3072,
               dropout: int = 0.1):
    super().__init__()

    # Create the norm layer (LN)
    self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

    # Create the MLP
    self.mlp = nn.Sequential(
        nn.Linear(in_features=embedding_dim,
                  out_features=mlp_size),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features=mlp_size,
                  out_features=embedding_dim),
        nn.Dropout(p=dropout)
    )

  def forward(self, x):
    x = self.layer_norm(x)
    x = self.mlp(x)
    return x
    # return self.mlp(self.layer_norm(x)) # same as above


# Transformer Input Shape:
# Input shape: [B, S, D]
#   B = batch size
#   S = sequence length (e.g., number of tokens or patches)
#   D = embedding dimension (e.g., 768)

# In Transformers, the MLP acts independently on each token or patch vector.
# So it's like:

# for i in range(B):
#     for j in range(S):
#         x[i, j] = MLP(x[i, j])  # input: [D], output: [D]

# Or more compactly:
# MLP(x)  # input: [B, S, D] → output: [B, S, D]
# Because PyTorch linear layers (nn.Linear) apply across the last dimension, you don’t need to reshape —
# it automatically applies to each [D] dimensional  vector independently.
# It's equivalent to a batched MLP over S × B vectors of shape [D]


# Visual Intuition:
# Think of x[i, j] as:
# i → the image or sentence in the batch
# j → the j-th patch or word (token)
# x[i, j] → the 768-dim vector for that token
# The MLP is applied to each token's vector, independent of others, just like in a regular MLP.
# It does not mix tokens or attend across time/space — that's what self-attention layers do.

# Typical Transformer block structure:
# Input → LayerNorm → Multi-head Self-Attention → Add & Norm → MLP → Add & Norm
# So the MLP operates after attention has already let tokens interact. It just updates each token’s features
# individually, kind of like:
# "Now that each token has talked to others, let me non-linearly transform its internal meaning."


# Summary
# In [B, S, D], the MLP works on the last dimension (D)
# It's applied independently to each token in the sequence
# Equivalent to a batched MLP over S × B vectors of shape [D]
# It's not a sequence model — it’s a feed-forward neural net per token


class TransformerEncoderBlock(nn.Module):
  def __init__(self,
               embedding_dim: int = 768,  # Hidden size D from table 1, 768 for ViT-Base
               num_heads: int = 12,  # from table 1
               mlp_size: int = 3072,  # from table 1
               mlp_dropout: int = 0.1,  # from table 3
               attn_dropout: int = 0):
    super().__init__()

    # Create MSA block (equation 2)
    self.msa_block = MultiHeadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                 num_heads=num_heads,
                                                 attn_dropout=attn_dropout)

    # Create MLP block (equation 3)
    self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                              mlp_size=mlp_size,
                              dropout=mlp_dropout)

  def forward(self, x):
    x = self.msa_block(x) + x  # residual/skip connection for equation 2
    x = self.mlp_block(x) + x  # residual/skip connection for equation 3
    return x


# Create a ViT class
class ViT(nn.Module):
  def __init__(self,
               img_size: int = 224,  # Table 3 from the ViT paper
               in_channels: int = 3,
               patch_size: int = 16,
               num_transformer_layers: int = 12,  # Table 1 for "Layers" for ViT-Base
               embedding_dim: int = 768,  # Hidden size D from Table 1 for ViT-Base
               mlp_size: int = 3072,  # Table 1
               num_heads: int = 12,  # Table 1
               attn_dropout: int = 0,
               mlp_dropout: int = 0.1,
               embedding_dropout: int = 0.1,  # Dropout for patch and position embeddings
               num_classes: int = 1000):  # number of classes in our classification problem
    super().__init__()

    # Make an assertion that the image size is compatible with the patch size
    assert img_size % patch_size == 0,  f"Image size must be divisible by patch size, image: {img_size}, patch size: {patch_size}"

    # Calculate the number of patches (height * width/patch^2)
    self.num_patches = (img_size * img_size) // patch_size**2

    # Create learnable class embedding (needs to go at front of sequence of patch embeddings)
    # class token embedding is a single fixed learnable vector, shared across all images, of the same dimension
    # as patch embeddings (e.g., 768).
    # he image is split into patches (e.g., 16x16). Each patch is linearly projected
    # to a 768-dimensional embedding. Then, a special [CLS] token (also 768-dimensional) is prepended to
    # the sequence of patch embeddings of each image  before feeding into the Transformer.
    # The [CLS] token is not unique per image. It starts as the same learnable parameter.
    # Its representation evolves per image through the attention layers, as it attends to all patches.
    # By the end of the transformer, the [CLS] token's output embedding becomes a summary representation of
    # the image, used for classification.
    self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                        requires_grad=True)

    # Create learnable position embedding of shape (1, number_of_patches+1, embedding_dim)
    self.position_embedding = nn.Parameter(
        data=torch.randn(1, self.num_patches+1, embedding_dim))

    # Create embedding dropout value
    self.embedding_dropout = nn.Dropout(p=embedding_dropout)

    # Create patch embedding layer
    self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                          patch_size=patch_size,
                                          embedding_dim=embedding_dim)

    # Create the Transformer Encoder block
    self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                       num_heads=num_heads,
                                                                       mlp_size=mlp_size,
                                                                       mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])

    # Create classifier head
    self.classifier = nn.Sequential(
        nn.LayerNorm(normalized_shape=embedding_dim),
        nn.Linear(in_features=embedding_dim,
                  out_features=num_classes)
    )

  def forward(self, x):
    # Get the batch size
    batch_size = x.shape[0]

    # Create class token embedding and expand it to match the batch size (equation 1)
    # self.class_embedding.shape = (1, 1, embedding_dim) = (1, 1, 768)
    # What does .expand(B, -1, -1) do?
    # It repeats the same [CLS] token across the batch without allocating new memory
    # (i.e., it's still the same tensor broadcasted across batch).
    # -1 means "keep the original size of this dimension"
    # cls_token.expand(32, -1, -1).shape == (32, 1, 768)
    class_token = self.class_embedding.expand(
        batch_size, -1, -1)  # "-1" means to infer the dimensions

    # Create the patch embedding (equation 1)
    x = self.patch_embedding(x)

    # Concat class token embedding and patch embedding (equation 1)
    # (batch_size, number_of_patches, embedding_dim)
    x = torch.cat((class_token, x), dim=1)

    # Add position embedding to class token and patch embedding
    x = self.position_embedding + x

    # Apply dropout to patch embedding ("directly after adding positional- to patch embeddings")
    x = self.embedding_dropout(x)
    # x.shape = (batch_size, number_of_patches+1, embedding_dim) = torch.Size([32, 197, 768])

    # Put 0th index logit through classifier (equation 4)
    # print(x[:, 0, :].shape) # torch.Size([32, 768])
    # get the class token embedding (the first token in the sequence for that image)
    x = self.classifier(x[:, 0, :])
    # print(x.shape)  # (batch_size, num_classes) = torch.Size([32, 3])

    return x
