import torch
from datasets import load_dataset
from torch.utils.data import random_split, DataLoader, Dataset

from tokenizer import get_or_build_tokenizer


class TranslationDataset(Dataset):
    def __init__(self, ds, src_tokenizer, out_tokenizer, src_lang="en", out_lang="it", seq_len=128) -> None:
        super().__init__()

        self.ds = ds
        self.src_tokenizer = src_tokenizer
        self.out_tokenizer = out_tokenizer
        self.src_lang = src_lang
        self.out_lang = out_lang
        self.seq_len = seq_len



        self.sos_token = torch.tensor(
            [src_tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor(
            [src_tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor(
            [src_tokenizer.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        out_text = src_target_pair['translation'][self.out_lang]

        src_tokens = self.src_tokenizer.encode(src_text).ids
        out_tokens = self.out_tokenizer.encode(out_text).ids

        src_num_pad = self.seq_len - len(src_tokens) - 2 # 2 for sos and eos
        out_num_pad = self.seq_len - len(out_tokens) - 1 # 1 for sos because for training we only add sos at the start of decoder input
        # and for the label we add eos at the end.

        if src_num_pad < 0 or out_num_pad < 0:
            raise ValueError(f"Sentence is too long. Exceeded max length of {self.seq_len} tokens by {abs(src_num_pad) if src_num_pad < 0 else abs(out_num_pad)}\n")


        # Input to the encoder is always [sos] + src_tokens + [eos] + [pad]*src_num_pad
        encoder_tokens = torch.concat(
            [
                self.sos_token,
                torch.tensor(src_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*src_num_pad, dtype=torch.int64)
            ]
        )

        # Input to the decoder is always [sos] + out_tokens + [pad]*out_num_pad
        # Label is always shifted one step right compared to the decoder input.
        # This is known as teacher forcing — during training, we give the decoder the correct previous token,
        # not its own previous prediction and expect it to predict the next token. And then to calculate the loss,
        # we compare the decoder output with the label which is one step ahead of the decoder input i,e the
        # next token in the sequence.
        decoder_tokens = torch.concat(
            [
                self.sos_token,
                torch.tensor(out_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token]*out_num_pad, dtype=torch.int64)
            ]
        )


        # Labels are always out_tokens + [eos] + [pad]*out_num_pad
        # Label is what we expect the decoder to output.
        labels = torch.concat(
            [
                torch.tensor(out_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*out_num_pad, dtype=torch.int64)
            ]
        )

        # encoder_input = [sos] + src_tokens + [eos] + [pad]*src_num_pad
        # decoder_input = [sos] + out_tokens + [pad]*out_num_pad
        # label = out_tokens + [eos] + [pad]*out_num_pad

        return {
            'encoder_input': encoder_tokens,  # seq_len
            'decoder_input': decoder_tokens,  # seq_len

            # [1,1, seq_len] -> broadcasts to [h,len, seq_len] in attention
            'encoder_mask': (encoder_tokens != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # [1,1,seq_len]

            # [1, seq_len, seq_len]  # (1, 1, seq_len) & (1, seq_len, seq_len)
            # First mask out the padding tokens in the decoder input so there is 1 where the decoder can attend
            # to the input tokens and 0 where it cannot.
            # Then apply the causal mask so that the decoder can only attend to previous tokens.
            'decoder_mask': (decoder_tokens != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(self.seq_len),

            'labels': labels,    # seq_len
            "src_text": src_text,
            "tgt_text": out_text,
        }


# def causal_mask(size):
#     mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
#     return mask == 0

def causal_mask(size):
    """
    Creates a causal mask for the decoder.
    The mask is a square matrix of size (1, len, len) where the upper triangular part is 1 and the lower triangular part is 0.
    This ensures that the decoder can only attend to previous tokens and not future tokens.

    torch.tril(torch.ones(1, 5, 5)) creates a lower triangular matrix of ones:
      [[[1, 0, 0, 0, 0],
      [1, 1, 0, 0, 0],
      [1, 1, 1, 0, 0],
      [1, 1, 1, 1, 0],
      [1, 1, 1, 1, 1]]]

    """
    mask = torch.tril(torch.ones(1, size, size)).type(torch.int)
    return mask == 1  # Returns a boolean mask where 1 is True and 0 is False




def get_dataset(config):
    # It only has the train split, so we divide it overselves
    # ds_raw = load_dataset("opus_books", "en-it", split='train')
    ds_raw = load_dataset( f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_target = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(
            item['translation'][config['lang_src']]).ids
        target_ids = tokenizer_target.encode(
            item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(target_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size

    train_ds_raw, val_ds_raw = random_split(
        ds_raw, [train_ds_size, val_ds_size])

    train_ds = TranslationDataset(train_ds_raw, tokenizer_src, tokenizer_target,
                                  config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = TranslationDataset(val_ds_raw, tokenizer_src, tokenizer_target,
                                config['lang_src'], config['lang_tgt'], config['seq_len'])

    train_dataloader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_target
