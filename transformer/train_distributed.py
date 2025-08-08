
# Data Parallelism vs Model Parallelism

# In PyTorch, a device is where computations occur, and data resides. The CPU and the GPU are examples of devices.
# A PyTorch tensor resides in a device, and its operations are executed on the same device.

# If the model can fit within a single GPU, then we can distribute the training on multiple servers(each containing
# one or multiple GPUs), with each GPU processing a subset of the entire dataset in parallel and synchronizing the
# gradients during backpropagation. This option is known as Data Parallelism.


# If the model cannot fit within a single GPU, then we need to “break” the model into smaller layers and let each GPU
# process a part of the forward/backward step during gradient descent. This option is known as Model Parallelism.


# Distributed Data Parallel in detail:

# Imagine you have a training script that is running on a single computer/GPU, but it’s very slow, because:
# the dataset is big and you can’t use a big batch size as it will result in an Out Of Memory error on CUDA.
# Distributed Data Parallel is the solution in this case.

# I will use the term “node” and “GPU” interchangeably. If a cluster is made up of 2 computers, each having 2 GPUs,
# then we have 4 nodes in total.
# Distributed Data Parallel works in the following way:
# 1. At the beginning of the training, the model’s weights are initialized on one node and sent to all the
#   other nodes(Broadcast)
# 2. Each node trains the same model(with the same initial weights) on a subset of the dataset.
#    Each node runs a forward and backward step on one or more batch of data. This will result in a local gradient.
#    The local gradient may be the accumulation of one or more batches.
# 3. Every few batches, the gradients of each node are accumulated on one node(summed up). The sum of all
#   the gradients is cumulated on one node(Reduce)
# 4. The cumulative gradient is sent to all the other nodes(Broadcast).
#    The sequence of Reduce and Broadcast are implemented as a single operation(All-Reduce).
# 4. Each node updates the parameters of its local model with the summed gradients received using its own optimizer.
#    After the update, the gradients are reset to zero and we can start another loop.
# 5. Go back to step 2


# Collective Communication Primitives:

# In distributed computing environments, a node may need to communicate with other nodes. If the communication pattern
# is similar to a client and a server, then we talk about point-to-point communication, because one client connects
# to one server in a requestresponse chain of events. However, there are cases in which one node needs to communicate
# to multiple receivers at once: this is the typical case of data parallel training in deep learning: one node needs
# to send the initial weights to all the other nodes. Moreover, all the other nodes, need to send their gradients to
# one single node for summation and receive back the cumulative gradient. Collective communication allows to model
# the communication pattern between groups of nodes.
# The operation of sending a data to all the other nodes is known as the Broadcast operator. Collective Communication
# libraries(e.g. NCCL) assign a unique ID to each node, known as RANK.

# Reduce operation:
# The Broadcast operator is used to send the initial weights to all the other nodes when we start the training loop.
# At every few batches of data processed by each node, the gradients of all nodes need to be sent to one node and accumulated
# (summed up). This operation is known as Reduce.

# Collective Communication: All-Reduce:

# Having accumulated the gradients of all the nodes into a single node, we need to send the cumulative gradient
# to all the nodes. This operation can be done using a Broadcast operator.
# The sequence of Reduce-Broadcast is implemented by another operator known as All-Reduce, whose runtime is generally lower
# than the sequence of Reduce followed by a Broadcast. You can think of it as a sequence of Reduce followed by a Broadcast
# operation.


# Failover: what happens if one node crashes:

# Imagine you’re training in a distributed scenario and one of the nodes suddenly crashes. In these case, 2
# GPUs out of 4 become unreachable. How should the system react?
# One way, would be to restart the entire cluster and that’s easy. However, by restarting the cluster,
# the training would restart from zero, and we would lose all the parameters and computation done so far.
# A better approach is to use checkpointing. Checkpointing means saving the weights of the model on a shared
# disk every few iterations(for example every epoch) and resume the training from the last checkpoint in case there’s a crash.


# When does PyTorch synchronize gradients?

# PyTorch will synchronize the gradients automatically every time we call the method loss.backward(). This will lead to:

# 1. Each node calculating its local gradients(derivative of the loss function w.r.t each node of the computational graph)
# 2. Each node will send its local gradient to one single node and receives back the cumulative gradient(All-Reduce)
# 3. Each node will update its weights using the cumulative gradient and its local optimizer.

# We can avoid PyTorch synchronizing the gradient at every backward step and instead, let it accumulate the gradient
# for a few steps by using the no_sync() context.


# PyTorch tricks: Computation-Communication overlap:

# Since each GPU needs to send its gradient to a central node for accumulation, this can lead to an idle time in which
# the GPUs are not working, but only communicating with each other. PyTorch handles this communication delay in a
# smart way by allowing the computation and communication to overlap. i.e., while the backward pass is still computing
# gradients for some layers from right to left, PyTorch can start communicating (synchronizing) the gradients
# of earlier layers in the background. This reduces idle time and improves overall training efficiency.

# PyTorch DDP Bucketing:

# Another trick to improve Computation-Communication overlap is "bucketing".
# Instead of synchronizing gradients for each layer as soon as they are ready,
# PyTorch groups gradients from multiple layers into "buckets" of a certain size (bucket_cap_mb).
# Once a bucket is full, its gradients are synchronized across GPUs.
# This reduces the number of communication calls and allows for better overlap between computation and communication,
# especially for models with many small layers. PyTorch recommends 25MB as the size of the bucket.
# You can control the bucket size in DistributedDataParallel via the bucket_cap_mb argument.


# In this section, we will look at the most basic case of distributed training: PyTorch’s DistributedDataParallel (DDP)
# strategy. DDP enables parallelism by splitting the input data across the available devices and processing these
# data subsets simultaneously.
# How does this work? PyTorch launches a separate process on each GPU, and each process receives and keeps a
# copy of the model – these copies will be synchronized during training.
# For more advanced use cases where a single model doesn’t fit onto the GPU, use PyTorch’s Fully Sharded Data Parallel (FSDP)
# method, which performs distributed data parallelism and distributes large layers across different GPUs.

# Since each model copy will see a different sample of the training data, the model copies will return different
# logits as outputs and compute different gradients during the backward pass. These gradients are then averaged
# and synchronized during training to update the models. This way, we ensure that the models don’t diverge


# Multi-GPU computing in interactive environments DDP does not function properly within interactive Python
# environments like Jupyter notebooks, which don’t handle multiprocessing in the same way a standalone Python script
# does. Therefore, the following code should be executed as a script, not within a notebook interface like Jupyter.
# This is because DDP needs to spawn multiple processes, and each process should have its own Python interpreter instance.



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Distributed training
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

import warnings
from tqdm import tqdm
import os
from pathlib import Path
import argparse

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import wandb
import torchmetrics

from model import get_model
from dataset import TranslationDataset, causal_mask
from config_distributed import get_default_config, get_weights_file_path, get_latest_weights_file_path, ModelConfig


def greedy_decode(model: nn.Module, source: torch.Tensor, source_mask: torch.Tensor, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, max_len: int, device: torch.device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.module.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(
            1)).type_as(source_mask).to(device)

        # calculate output
        # When using Distributed Training instead of calling model.encode or model.any_method(),
        # we use model.module.decode() or model.module.any_method()
        out = model.module.decode(
            encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.module.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model: nn.Module, validation_ds: DataLoader, tokenizer_src: Tokenizer, tokenizer_tgt: Tokenizer, max_len: int, device: torch.device, print_msg: callable, global_step: int, num_examples: int = 2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(
                device)  # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(
                model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(
                model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break

    # Evaluate the character error rate
    # Compute the char error rate
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    wandb.log({'validation/cer': cer, 'global_step': global_step})

    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    wandb.log({'validation/wer': wer, 'global_step': global_step})

    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    wandb.log({'validation/BLEU': bleu, 'global_step': global_step})


def get_all_sentences(ds: Dataset, lang: str):
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config: ModelConfig, ds: Dataset, lang: str) -> Tokenizer:
    tokenizer_path = Path(config.tokenizer_file.format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset(config: ModelConfig):
    # It only has the train split, so we divide it overselves
    ds_raw = load_dataset(
        'opus_books', f"{config.lang_src}-{config.lang_tgt}", split='train')

    # Build tokenizers
    print(f"GPU {config.local_rank} - Loading tokenizers...")
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config.lang_src)
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config.lang_tgt)

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(
        ds_raw, [train_ds_size, val_ds_size])

    train_ds = TranslationDataset(train_ds_raw, tokenizer_src, tokenizer_tgt,
                                config.lang_src, config.lang_tgt, config.seq_len)
    val_ds = TranslationDataset(val_ds_raw, tokenizer_src, tokenizer_tgt,
                              config.lang_src, config.lang_tgt, config.seq_len)

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(
            item['translation'][config.lang_src]).ids
        tgt_ids = tokenizer_tgt.encode(
            item['translation'][config.lang_tgt]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'GPU {config.local_rank} - Max length of source sentence: {max_len_src}')
    print(f'GPU {config.local_rank} - Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config.batch_size,
                                  shuffle=False,  # for distributed training set shuffle to False
                                  # DistributedSampler will handle the shuffling
                                  # and splitting of the dataset across the GPUs without overlapping samples.
                                  sampler=DistributedSampler(train_ds, shuffle=True))
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def train_model(config: ModelConfig):
    # Define the device
    assert torch.cuda.is_available(), "Training on CPU is not supported"
    device = torch.device("cuda")
    print(f"GPU {config.local_rank} - Using device: {device}")

    # Make sure the weights folder exists
    Path(config.model_folder).mkdir(parents=True, exist_ok=True)

    # Load the dataset
    print(f"GPU {config.local_rank} - Loading dataset...")
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(
        config)
    model = get_model(config, tokenizer_src.get_vocab_size(),
                      tokenizer_tgt.get_vocab_size()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-9)

    # By default, load the latest checkpoint
    initial_epoch = 0
    global_step = 0
    wandb_run_id = None
    if config.preload != '':

        if config.preload == 'latest':
            # Get the filename of the latest checkpoint
            model_filename = get_latest_weights_file_path(config)
        else:
            # In case we want to preload a specific checkpoint
            model_filename = get_weights_file_path(config, int(config.preload))

        if model_filename is not None:
            print(f'GPU {config.local_rank} - Preloading model {model_filename}')
            state = torch.load(model_filename)
            model.load_state_dict(state['model_state_dict'])
            initial_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']
            wandb_run_id = state['wandb_run_id']
            del state
        else:
            # If we couldn't find a model to preload, just start from scratch
            print(
                f'GPU {config.local_rank} - Could not find model to preload: {config.preload}. Starting from scratch')

    # Only initialize W&B on the global rank 0 node
    if config.local_rank == 0:
        wandb.init(
            # set the wandb project where this run will be logged
            project="pytorch-transformer-distributed",
            # allow resuming existing run with the same name (in case the rank 0 node crashed)
            name=f"global_rank_{config.global_rank}",
            id=wandb_run_id,
            resume="allow",
            group=config.wandb_group,
            # track hyperparameters and run metadata
            config=config
        )

    # Convert the model to DistributedDataParallel
    # Here we can also specify the bucket_cap_mb parameter to control the size of the buckets
    # the core model is now accessible as model.module
    model = DistributedDataParallel(model, device_ids=[config.local_rank])

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id(
        '[PAD]'), label_smoothing=0.1).to(device)

    if config.global_rank == 0:
        # define our custom x axis metric
        wandb.define_metric("global_step")
        # define which metrics will be plotted against it
        wandb.define_metric("validation/*", step_metric="global_step")
        wandb.define_metric("train/*", step_metric="global_step")

    for epoch in range(initial_epoch, config.num_epochs):
        torch.cuda.empty_cache()
        model.train()

        # Disable tqdm on all nodes except the rank 0 GPU on each server
        batch_iterator = tqdm(
            train_dataloader, desc=f"Processing Epoch {epoch:02d} on rank {config.global_rank}", disable=config.local_rank != 0)

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(
                device)  # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(
                device)  # (B, 1, seq_len, seq_len)

            # # Run the tensors through the encoder, decoder and the projection layer
            # encoder_output = model.module.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            # decoder_output = model.module.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            # proj_output = model.module.project(decoder_output) # (B, seq_len, vocab_size)
            proj_output = model(encoder_input, encoder_mask,
                                decoder_input, decoder_mask)

            # Compare the output with the label
            label = batch['label'].to(device)  # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix(
                {"loss": f"{loss.item():6.3f}", "global_step": global_step})

            if config.local_rank == 0:
                # Log the loss
                wandb.log({'train/loss': loss.item(),
                          'global_step': global_step})

            # Backpropagate the loss
            # When we call loss.backward() in a Distributed training setting, PyTorch will automatically
            # synchronize gradients across all the GPUs for us.
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Only run validation and checkpoint saving on the rank 0 node
        if config.global_rank == 0:
            # Run validation at the end of every epoch
            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt,
                           config.seq_len, device, lambda msg: batch_iterator.write(msg), global_step)

            # Save the model at the end of every epoch
            model_filename = get_weights_file_path(config, epoch)
            torch.save({
                'epoch': epoch,
                # Need to access module because we are using DDP
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'wandb_run_id': wandb.run.id  # Save to resume logging data
            }, model_filename)


#  We have a __name__ == "__main__" clause at the bottom of the script that is executed when we run the
# file as a standalone Python script (as opposed to importing it as a module)
if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # Disable tokenizers parallelism (this is to avoid deadlocks when creating the tokenizers on multiple GPUs)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = get_default_config()

    # Read command line arguments and overwrite config accordingly
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=config.batch_size)
    parser.add_argument('--num_epochs', type=int, default=config.num_epochs)
    parser.add_argument('--lr', type=float, default=config.lr)
    parser.add_argument('--seq_len', type=int, default=config.seq_len)
    parser.add_argument('--d_model', type=int, default=config.d_model)
    parser.add_argument('--lang_src', type=str, default=config.lang_src)
    parser.add_argument('--lang_tgt', type=str, default=config.lang_tgt)
    parser.add_argument('--model_folder', type=str,
                        default=config.model_folder)
    parser.add_argument('--model_basename', type=str,
                        default=config.model_basename)
    parser.add_argument('--preload', type=str, default=config.preload)
    parser.add_argument('--tokenizer_file', type=str,
                        default=config.tokenizer_file)
    parser.add_argument('--wandb_group', type=str, default="exp1")
    args = parser.parse_args()

    # Update default configuration with command line arguments
    config.__dict__.update(vars(args))

    # Add local rank and global rank to the config
    # Local Rank is the rank of the process or GPU on local computer. Say
    # you have two computers with 3 GPUs each, then the local rank will be 0, 1, 2 on the first computer
    # and 0, 1, 2 on the second computer. And the global rank will be 0, 1, 2, 3, 4, 5 across both computers.
    # Global Rank is the rank of the process or GPU across all computers. Global Rank indicates a globally
    # unique identifier among all the gpus in the cluster of distributed training setup.
    config.local_rank = int(os.environ['LOCAL_RANK'])
    config.global_rank = int(os.environ['RANK'])

    # number of processes in the distributed run
    ddp_world_size = int(os.environ['WORLD_SIZE'])

    assert config.local_rank != -1, "LOCAL_RANK environment variable not set"
    assert config.global_rank != -1, "RANK environment variable not set"

    # Print configuration (only once per server)
    if config.local_rank == 0:
        print("Configuration:")
        for key, value in config.__dict__.items():
            print(f"{key:>20}: {value}")

    # Only print on global_rank 0 to avoid duplicate prints from each GPU process
    if config.global_rank == 0:
      print("Number of GPUs available:", torch.cuda.device_count())

    # Setup distributed training
    # At this line, torchrun will stop and wait for all the nodes to connect
    # This is done to ensure that all the nodes are ready before starting the training
    # The init_process_group function should be called at the beginning of the training script to initialize a
    # process group for each process in the distributed setup
    # Windows users may have to use "gloo" instead of "nccl" as backend
    # gloo: Facebook Collective Communication Library
    # init_process_group(backend="gloo")
    init_process_group(backend='nccl')

    # to tell the cuda which GPU to use for this process
    # This sets the device having a specific rank for the current process using the provided rank.
    torch.cuda.set_device(config.local_rank)

    # Train the model
    train_model(config)

    # Clean up distributed training
    destroy_process_group()


# When running the script using torchrun, it automatically launches one process per GPU and assigns each
# process a unique rank, along with other distributed training metadata (like world size and local rank),
# which are passed into the script via environment variables. In the __main__ block, we read these variables
# using os.environ


# torchrun --nproc_per_node=2 --nnodes=2 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=IP_ADDR_MASTER_NODE:48123
# train.py --batch_size 8 --model_folder "/mnt/training-data/weights"