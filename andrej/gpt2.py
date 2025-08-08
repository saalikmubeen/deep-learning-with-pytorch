import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import numpy as np
import tiktoken
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
# -----------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        # assert that the embedding dimension is divisible by the number of heads
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # register a buffer to store the lower triangular matrix for causal self attention
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C //
                   self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self attention
        # attn = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1)))
        # attn = attn.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        # attn = F.softmax(attn, dim=-1)
        # y = attn @ v # (B, nh, T, T) @ (B, nh, T, hs) = (B, nh, T, hs)
        # y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # ? kernel fusion. flash attention is a kernel fusion algorithm, but
        # torch.compile cannot find all the optimizations required to do attention in an efficient way
        # like flash attention by default because it requires an algorithmic rewrite of how the attention
        # is implemented. That's why flash attention was developed to do this in a more efficient way
        # by fusing the kernel operations together. Flash attention is just an algorithmic rewrite.
        # Flash attention does more number of FLOPs than the original attention, but it is more efficient
        # and faster(around 7.6x) because it is very mindful of the memory hierarchy and how the data is accessed
        # in the memory. It uses the memory more efficiently and does not require a lot of memory bandwidth
        # to do the attention computation.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True)  # flash attention
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.c_proj(y)
        return y


class TanhGELU(nn.Module):

    def forward(self, x):
      return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')  # Gaussian Error Linear Unit
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        # In original attention paper, they use layer normalization after the attention layer
        # but in the GPT-2 paper, they use layer normalization before the attention layer
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # residual connection
        # Similarly, in the original attention paper, they use layer normalization after the MLP layer
        # but in the GPT-2 paper, they use layer normalization before the MLP or feedforward layer
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    # max sequence length for the model to process at once i.e context length
    block_size: int = 1024
    # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    vocab_size: int = 50257
    n_layer: int = 12  # number of layers or blocks in the transformer
    n_head: int = 12  # number of attention heads in each block or layer for multi-head attention
    n_embd: int = 768  # embedding dimension or the number of dimensions to represent each token in the embedding space


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ModuleDict allows us to store multiple sub modules in a single container
        # and access and index them by name like a dictionary
        # e.g. self.transformer['wte'] is the word token embedding layer
        # transformer.wte.weight ––> torch.Size([50257, 768])
        # transformer is the main container.

        # ModuleList allows us to store multiple modules in a single container
        # and access and index them by index like a list
        # e.g. self.transformer[0] is the first block or first layer of the transformer
        # and self.transformer.1 is the second block or second layer of the transformer
        # transformer.h.0.ln_1.weight ––> torch.Size([768])
        # transformer.h[0].attn.c_attn.weight ––> torch.Size([768, 768])

        self.transformer = nn.ModuleDict(dict(
            # token embedding layer
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            # position embedding layer
            wpe=nn.Embedding(config.block_size, config.n_embd),

            # We will have 12 blocks or layers in the transformer
            # list of blocks or layers
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),  # final layer norm
        ))

        # final classifier layer
        # linear layer to project the embedding to the vocabulary size
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing or tying scheme
        # the weight matrix of the token embedding layer of shape (vocab_size, n_embd)
        # is shared with the weight matrix of the final linear layer of shape (n_embd, vocab_size)
        # this means the same weights are used for both the token embeddings and the final logits
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        # self.apply is a method that iterates and applies a function to all the sub modules
        # in this module model
        # here we apply the _init_weights function to all the modules in the model
        # _init_weights initializes the weights of the model according to the GPT-2 paper
        # the weights are initialized with a normal distribution with mean 0 and std 0.02
        self.apply(self._init_weights)

    # GPT-2 paper uses a specific initialization scheme for the weights
    # this function is called by self.apply to initialize the weights of the model
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long,
                           device=idx.device)  # shape (T)
        # position embeddings of shape (T, n_embd)
        pos_emb = self.transformer.wpe(pos)
        # token embeddings of shape (B, T, n_embd)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)

        # ? If the input was (B, T) indicies, then at every single (B, T) we calculate logits(vocab_size)
        # ? for what token comes next in the sequence.
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # ? This is used to debug the model and inspect the logits
        # So when we run the file, we can inspect the logits and the loss in the interactive console
        # that will be launched when we hit the below line.
        # import code; code.interact(local=locals())
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
            # OR
            # loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T), ignore_index=-1)
        return logits, loss

    @classmethod
    # This method is used to load pretrained GPT-2 model weights from Hugging Face
    # We can load the weights from the Hugging Face model hub
    # and use them to initialize our own gpt2 model
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            # 124M params
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
            # 350M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
            # 774M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
            # 1558M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        # always 50257 for GPT model checkpoints
        config_args['vocab_size'] = 50257
        # always 1024 for GPT model checkpoints
        config_args['block_size'] = 1024
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)  # ? Our own GPT2 model
        # ? Our own GPT2 model state dictionary containing all the parameters
        sd = model.state_dict()
        sd_keys = sd.keys()
        # discard this mask / buffer, not a param
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # init a huggingface/transformers model # ? Pretrained GPT2 model from Hugging Face
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        # ? Pretrained GPT2 model state dictionary containing all the parameters
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        # from the huggingface model, copy the weights to our own model
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(
            '.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(
            '.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
                      'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them

        # assert that the number of keys in the state dictionary of the pretrained model is the same as the
        # number of keys in the state dictionary of our own model, i.e our own model matches the architecture
        # of the huggingface model
        assert len(sd_keys_hf) == len(
            sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            # if the key ends with any of the strings in the transposed list
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose

                # checks if the shape of the key in the pretrained model is the same as the shape of the key in our own model
                # sd_hf[k].shape[::-1] reverses the shape of the key in the pretrained model because the weights are transposed
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            # default weight decay used in AdamW is 0.01, but we use 0.1 as in the GPT-2 paper
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(
            torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        # fused=use_fused is a flag that enables the kernel fused version of AdamW optimizer update.
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(
            0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


# -----------------------------------------------------------------------------


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)  # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


# class DataLoaderLite:
#     def __init__(self, B, T, process_rank, num_processes):
#         self.B = B
#         self.T = T
#         self.process_rank = process_rank
#         self.num_processes = num_processes

#         # at init load tokens from disk and store them in memory
#         with open('input.txt', 'r') as f:
#             text = f.read()
#         enc = tiktoken.get_encoding('gpt2')
#         tokens = enc.encode(text)
#         self.tokens = torch.tensor(tokens)
#         if master_process:
#             print(f"loaded {len(self.tokens)} tokens")

#         # state
#         # process 0 starts at position 0, process 1 starts at B * T, etc.
#         self.current_position = self.B * self.T * self.process_rank

#     def next_batch(self):
#         B, T = self.B, self.T
#         buf = self.tokens[self.current_position: self.current_position+B*T+1]
#         x = (buf[:-1]).view(B, T)  # inputs
#         y = (buf[1:]).view(B, T)  # targets
#         # advance the position in the tensor
#         self.current_position += B * T * self.num_processes
#         # if loading the next batch would be out of bounds, reset
#         if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
#             self.current_position = self.B * self.T * self.process_rank
#         return x, y


# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    # we must shift mask, so we start at the last prompt token
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py
# Now instead of using python train_gpt2.py, we use torchrun to launch the script
# which will create a distributed environment and span 8 processes, one for each GPU
# and set the environment variables RANK, LOCAL_RANK, and WORLD_SIZE


# run the training loop

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run or simple run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    # all processes have a unique rank in the distributed run and that's the way for
    # us to coordinate between them so they don't run on the same data and run on different
    # parts of the data.
    ddp_rank = int(os.environ['RANK'])

    # only used in a multi-node setting (we currently have a single node with 8 GPUs)
    # LOCAL_RANK is the rank of the process on the local node, i.e. the GPU number (0-7)
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    # number of processes in the distributed run
    ddp_world_size = int(os.environ['WORLD_SIZE'])

    # which GPU to use for this process. Depending on the LOCAL_RANK of this process it's assigned a GPU
    # so that there is no collision on which GPU is used by which process.
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    # this process will do logging, checkpointing etc.
    master_process = ddp_rank == 0
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")


# ? Gradient Accumulation
# A batch size of 0.5M tokens is too large to fit in memory, so we use gradient accumulation
#  524288 is the total batch size in number of tokens
# To get the total batch size in number of individual sequences, we divide 524288 by context length T
# If context length is 1024, then the total batch size in number of sequences is 524288 / 1024 = 512
total_batch_size = 524288  # 2**19, ~0.5M, in number of tokens
B = 64  # micro batch size.
T = 1024  # sequence length
assert total_batch_size % (
    B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
# grad_accum_steps is the number of steps we need to accumulate gradients before doing an optimizer step
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


# import sys; sys.exit(0) # exit early

train_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")


# torch.set_float32_matmul_precision('highest') # this is the default in PyTorch 2.0+ and the floats are set to the float32 data type

# ? this sets the datatype for matmul operations to TensorFloat32 on CUDA GPU devices
# The numbers are still stored in float32 in our code everywhere  but when they are transferred to the GPU
# for matmul operations, they are converted to TensorFloat32 and the matmul operations are done in TensorFloat32
# This does reduce the precision of the matmul operations, but it is still very accurate
torch.set_float32_matmul_precision('high')

# create model
# 50304 is a nicer vocabulary size than 50257 because 50304 is of form 2^n and is much
# nicer for hardware to work with, so we use it here
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
use_compile = False  # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
# always contains the "raw" unwrapped model
raw_model = model.module if ddp else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
# 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
max_steps = 19073


def get_lr(it):
    '''Learning rate scheduler: warmup + cosine decay'''

    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    # coeff starts at 1 and goes to 0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# optimize!
# optimizer = optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:  # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)

                # ? Quantization
                # ? torch.autocast is used to enable mixed precision training
                # it automatically casts the inputs to bfloat16 when the model is on a CUDA device
                # and the matrix multiplication is done in bfloat16 inside the GPUs. This speeds up training
                # and reduces memory usage, while still keeping the model in float32 precision.
                # Only the forwardpass and loss calculations should be in autocast,
                # the backward pass and optimizer step should not be in autocast.

                # unlike torch.set_float32_matmul_precision('high') which sets the precision for
                # all matmul operations only and does not affect the data type of the tensors outside GPU,
                # i.e if we check the data type of the tensors, they are still float32,
                # torch.autocast actually changes the data type of the tensors to bfloat16
                # and the matmul operations are done in bfloat16. So this change is not just local to the operation
                # itself. If we use float16 instead of bfloat16, we will need gradient scaling
                # to avoid underflow and overflow issues, but bfloat16 does not need gradient scaling
                # and is more stable for training large models.

                # float32 -> 1 signed bit + 8 exponent bits + 23 mantissa bits (4 bytees of memory)

                # float16 -> 1 signed bit + 5 exponent bits + 10 mantissa bits (2 bytes of memory)
                # float16 has lower precision than float32 as well as lower range(because of fewer exponent bits),
                # so it can represent smaller and larger numbers than float16.

                # bfloat16 -> 1 signed bit + 8 exponent bits + 7 mantissa bits (2 bytes of memory)
                # bfloat16 is a hybrid between full precision float32 and half precision float16,
                # bfloat16 is a truncated version of float32, it has the same exponent bits(i.e same range)
                # but fewer mantissa bits, so it has lower precision than float32,
                # but it has the same range as float32, so it can represent very large and very small numbers.

                # int8 -> 1 signed bit + 7 fraction bits for the value (1 byte of memory)

                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                    # print(logits.dtype) # torch.bfloat16 if using autocast
                    # print(model.transformer.wte.weight.dtype) # torch.float32, weights are still in float32
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(
                num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(
                num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(
                f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)  # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :]  # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(
                    topk_probs, 1, generator=sample_rng)  # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            #
            model.require_backward_grad_sync = (
                micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()

        # ? Different data is processed in parallel on different GPUs, and a separate forward and backward pass
        # is on each GPU. When the backward is called, and if the model.require_backward_grad_sync is True,
        # the gradients are synchronized across all processes, i.e. all GPUs.
        # This means that the gradients are averaged across all processes, i.e. all GPUs and the model
        # is updated on each GPU which is always identical across all GPUs.

        # There is another method called Fully Sharded Data Parallel (FSDP) which shards the model
        # across all GPUs and each GPU only has a part of the model, but this is not used here.

        # Once the forward pass is done, DPP will call all_reduce an basically it does an average
        # of the gradients across all processes, i.e. all GPUs and it will deposit that average
        # into the gradients of the model parameters on each GPU.
        loss.backward()  # deposits gradients(+=), gradients will add up on the gradient tensors over the micro steps
    if ddp:
        # all_reduces averages the loss_accum across all processes, i.e. all GPUs
        # this is necessary to ensure that the loss is the same across all processes
        # and that the gradients are averaged across all processes
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # ? clip the gradients to avoid exploding gradients
    # ? this is a common technique to stabilize training of large models
    # ? we clip the gradients to a maximum norm of 1.0
    # ? this means that the gradients are scaled down if their norm is larger than 1.0
    # ? What this function is doing is it calculates the global norm of all the gradients
    # ? of all the parameters in the model, i.e every single parameter that requires gradients
    # ? you square the gradients, sum them up, take the square root, and then you have the global norm,
    # ? that's the norm of the global gradient vector.
    # ? if the global norm is larger than 1.0, you scale down the gradients by a factor
    # ? of the global norm divided by 1.0, so that the global norm becomes 1.0.
    # ? Sometimes the batch might be unlucky and would produce bigger loss. Bigger loss means bigger gradients,
    # ? which can shock the model.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        # wait for the GPU to finish work and then measure the time, kinda like await in async code
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0  # time difference in seconds
    tokens_processed = train_loader.B * \
        train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()
