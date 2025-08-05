import math
import torch
import torch.nn as nn





class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    '''Positional Encoding for Transformer models.
      PE(pos, 2i) = sin(pos / (10000 ** (2i / d_model)))   for even indices in the embedding vector
      PE(pos, 2i+1) = cos(pos / (10000 ** (2i / d_model))) for odd indices in the embedding vector
      where:
          pos is the position of the token in the sequence,
          i is just an index over half of the embedding dimensions.
          For example, if your d_model = 512, then i goes from 0 to 255.
          Then 2i and 2i+1 give you the even and odd positions (i.e., indices 0 to 511).

      This class implements the positional encoding as described in the "Attention is All You Need" paper.
      The positional encoding is added to the input embeddings to give the model a sense of the order of the tokens in the sequence.
      Args:
          d_model (int): The dimension of the model (i.e., the size of the embeddings).
          seq_len (int): The length of the input sequences.
          dropout (float): Dropout rate to apply after adding positional encoding.
      Returns:
          A tensor of shape (batch_size, seq_len, d_model) with positional encodings added to the input embeddings.
      Reference:
          "Attention is All You Need" by Vaswani et al. (2017) - https://arxiv.org/abs/1706.03762
          Section 3.5: Positional Encoding
          https://arxiv.org/pdf/1706.03762.pdf#page=6
    '''

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # for pos in range(seq_len):
        #     for i in range(0, d_model, 2):  # step 2: even indices so we looping over 2i indicies directly
        #         angle = pos / (10000 ** (i / d_model))
        #         pe[pos, i] = math.sin(angle)      # even index → sin
        #         if i + 1 < d_model:
        #             pe[pos, i + 1] = math.cos(angle)  # odd index → cos

        # i = 0 → pe[pos, 0] = sin(...), pe[pos, 1] = cos(...)
        # i = 2 → pe[pos, 2] = sin(...), pe[pos, 3] = cos(...)
        # i = 4 → pe[pos, 4] = sin(...), pe[pos, 5] = cos(...)
        # i = 6 → pe[pos, 6] = sin(...), pe[pos, 7] = cos(...)


        # Create a vector of shape (seq_len)
        # Represents the position of each token in the sequence
        # Represents the numerator for the sine and cosine functions
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)

        # Create a vector of shape (d_model / 2)
        # Represents the denominator for the sine and cosine functions
        # denominator = torch.pow(10000, torch.arange(0, d_model, 2).float() / d_model) # shape = (d_model / 2)
        denominator = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model / 2)

        # Apply sine to even indices
        # sin(position * (10000 ** (2i / d_model))
        pe[:, 0::2] = torch.sin(position * denominator)

        # Apply cosine to odd indices
        # cos(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(position * denominator)

        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # Register the positional encoding as a buffer
        # This allows it to be part of the model's state but not a parameter that gets updated during training.
        # Buffers are tensors that are not considered model parameters but should be part of the model's state
        # and are saved in the model's state_dict when saving the model to disk.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # (batch, seq_len, d_model)
        pe = (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        x = x + pe
        return self.dropout(x)


class LayerNormalization(nn.Module):

    '''
      LayerNorm normalizes across the feature dimension, not the batch or sequence dimensions.
      It's applied independently to each token (i.e., each row in the sequence).
      LayerNorm normalizes across the hidden dimension, for each token independently.
    '''

    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps

        # These are broadcast across the batch and sequence dimensions, applying one scale and one
        # shift per hidden unit.

        # alpha is a learnable parameter
        self.alpha = nn.Parameter(torch.ones(features))  # (hidden_size)

        # bias is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features))  # (hidden_size)

    def forward(self, x):

        # x: (batch, seq_len, hidden_size)

        # If we want to apply BatchNorm
        # reshape to (batch * seq_len, hidden_size) to apply BatchNorm over batch dimension
        # b, s, h = x.shape
        # x_ = x.view(-1, h)  # (batch * seq_len, hidden_size)
        # mean = x_.mean(dim=0, keepdim=True)  # (batch * seq_len, 1)
        # std = x_.std(dim=0, keepdim=True)   # (batch * seq_len, 1)


        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1) will be broadcasted to (batch, seq_len, hidden_size)

        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1) will be broadcasted to (batch, seq_len, hidden_size)

        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        # return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x



class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h  # Dimension of vector seen by each head

        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq # query projection for x
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk # key projection for x
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv # value projection for x

        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo # output projection for x
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask is a lower triangular matrix of shape:
            # (batch, 1, seq_len, seq_len) for decoder to ensure that the decoder can only attend to previous tokens
            #        or
            # (batch, 1, 1, seq_len) for encoder (for encoder, mask puts 1 where the input tokens are not
            # padding tokens and 0 where they are padding to ensure that the model does not attend to padding tokens)
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)

        # (batch, h, seq_len, seq_len) # Apply softmax
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):

        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)


        # Reshape the query, key, and value tensors to split into multiple heads
        # so each head gets a portion of the d_model dimension.
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        # Transpose the dimensions to have (batch, h, seq_len, d_k)
        # Each head is now easily accessible along dimension 1
        # You can compute attention for each head independently and in parallel
        # You can think of it like this:
        # Before: (batch, seq_len, d_model) we were calculating attention for each example in the batch
        # independently. and now instead of having batch number of examples, that we calculate attention for,
        # we have batch * h number of examples (where h is the number of heads) that we calculate attention for
        # independently and in parallel.
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)

        # When you transpose or reshape a tensor, the new tensor may not be stored in a contiguous block of memory. That can cause issues with operations like .view(), which require contiguous memory layout.
        # What does .contiguous() do?
        # It returns a new tensor with the same data but stored in a contiguous chunk of memory.
        # It doesn't change the shape, but it makes the underlying memory layout continuous, which is necessary
        # for .view() to work correctly.
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    '''
      Residual connection with layer normalization.
      This is used to add the output of a sublayer (e.g., multi-head attention or feed-forward network)
      to the input of the sublayer, followed by layer normalization (in case of Attention is All You Need paper).
      The residual connection helps in training deep networks by allowing gradients to flow through the network
      without vanishing or exploding.
    '''

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        # In attention paper Norm is applied to output of the sublayer but here it is applied to the input
        # before it is passed to the sublayer.
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # Residual connection calls the sublayer (self_attention or feed_forward) and adds the result to the input x.
        # We just pass it as a lambda function to the ResidualConnection.
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention: MultiHeadAttention, cross_attention: MultiHeadAttention, feed_forward: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, decoder_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention(x, encoder_output, encoder_output, encoder_mask))
        x = self.residual_connections[2](x, self.feed_forward)
        return x


class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers # List of EncoderBlocks one after another
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers # List of DecoderBlocks one after another
        # Layer normalization is applied to the output of the last decoder block
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, encoder_mask, decoder_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, encoder_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, encoder_mask)

    def decode(self, encoder_output: torch.Tensor, encoder_mask: torch.Tensor, tgt: torch.Tensor, decoder_mask: torch.Tensor):
        # tgt.shape: (batch, seq_len)
        tgt = self.tgt_embed(tgt) # tgt.shape: (batch, seq_len, d_model)
        tgt = self.tgt_pos(tgt)   # tgt.shape: (batch, seq_len, d_model)
        return self.decoder(tgt, encoder_output, encoder_mask, decoder_mask)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int =512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(
            d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(
            d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(
            d_model, h, dropout)
        feed_forward_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block,
                                     decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len,
                              config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model
