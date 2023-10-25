import math
from timeit import default_timer as timer
from typing import Iterable, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator

torch.set_num_threads(1)

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

# common parameters
BATCH_SIZE = 128
# FFN_HID_DIM = 512
# NUM_ENCODER_LAYERS = 3
# NUM_DECODER_LAYERS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TranslationTransformer:
    def __init__(
        self,
        data_iterator=None,
        EMB_SIZE=16,
        FFN_HID_DIM=16,
        NUM_ENCODER_LAYERS=1,
        NUM_DECODER_LAYERS=1,
        NHEAD=4,
        learning_rate=0.0001,
    ):
        if data_iterator is None:
            SRC_LANGUAGE = "de"
            TGT_LANGUAGE = "en"

            def data_iterator(split="train"):
                train_iter = Multi30k(
                    split=split, language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)
                )
                return train_iter

        else:
            SRC_LANGUAGE = "data"
            TGT_LANGUAGE = "GP"
        self.SRC_LANGUAGE = SRC_LANGUAGE
        self.TGT_LANGUAGE = TGT_LANGUAGE
        self.get_data_iterator = data_iterator

        token_transform = {}
        vocab_transform = {}
        self.vocab_transform = vocab_transform
        self.classical_mode = SRC_LANGUAGE == "de" and TGT_LANGUAGE == "en"

        if self.classical_mode:
            token_transform[SRC_LANGUAGE] = get_tokenizer(
                "spacy", language="de_core_news_sm"
            )
            token_transform[TGT_LANGUAGE] = get_tokenizer(
                "spacy", language="en_core_web_sm"
            )
        else:
            token_transform[SRC_LANGUAGE] = get_tokenizer(None)
            token_transform[TGT_LANGUAGE] = get_tokenizer(None)

        def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
            language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

            for data_sample in data_iter:
                yield token_transform[language](data_sample[language_index[language]])

        if self.classical_mode:
            language_list = [SRC_LANGUAGE, TGT_LANGUAGE]
        else:
            language_list = [TGT_LANGUAGE]
        for ln in language_list:
            train_iter = self.get_data_iterator("train")
            vocab_transform[ln] = build_vocab_from_iterator(
                yield_tokens(train_iter, ln),
                min_freq=1,
                specials=special_symbols,
                special_first=True,
            )

        for ln in language_list:
            vocab_transform[ln].set_default_index(UNK_IDX)

        torch.manual_seed(0)

        if self.classical_mode:
            SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
        else:
            SRC_VOCAB_SIZE = 0
        TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

        transformer = Seq2SeqTransformer(
            NUM_ENCODER_LAYERS,
            NUM_DECODER_LAYERS,
            EMB_SIZE,
            NHEAD,
            SRC_VOCAB_SIZE,
            TGT_VOCAB_SIZE,
            FFN_HID_DIM,
            classical_mode=self.classical_mode,
        )
        self.transformer = transformer

        # parameter initialization
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        transformer = transformer.to(DEVICE)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.loss_fn = loss_fn
        optimizer = torch.optim.Adam(
            transformer.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9
        )
        self.optimizer = optimizer

        # text transformation tool
        text_transform = {}
        self.text_transform = text_transform
        if self.classical_mode:
            for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
                text_transform[ln] = sequential_transforms(
                    token_transform[ln], vocab_transform[ln], tensor_transform
                )
        else:
            text_transform[SRC_LANGUAGE] = lambda x: torch.tensor(x)
            ln = TGT_LANGUAGE
            text_transform[ln] = sequential_transforms(
                token_transform[ln], vocab_transform[ln], tensor_transform
            )

        def collate_fn(batch):
            src_batch, tgt_batch = [], []
            for src_sample, tgt_sample in batch:
                if self.classical_mode:
                    src_batch.append(
                        text_transform[SRC_LANGUAGE](src_sample.rstrip("\n"))
                    )
                else:
                    src_batch.append(text_transform[SRC_LANGUAGE](src_sample))
                tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

            src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
            tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
            return src_batch, tgt_batch

        self.collate_fn = collate_fn

    def train(self, NUM_EPOCHS=5):
        transformer = self.transformer
        train_loss = 0
        for epoch in range(1, NUM_EPOCHS + 1):
            start_time = timer()
            train_loss = self.train_epoch(transformer, self.optimizer, self.loss_fn)
            end_time = timer()
            val_loss = self.evaluate(transformer)
            # print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
            #        f""f"Epoch time = {(end_time - start_time):.3f}s"))
        return train_loss

    def train_epoch(self, model, optimizer, loss_fn):
        model.train()
        losses = 0
        train_iter = self.get_data_iterator("train")
        train_dataloader = DataLoader(
            train_iter, batch_size=BATCH_SIZE, collate_fn=self.collate_fn
        )

        for src, tgt in train_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src, tgt_input
            )

            logits = model(
                src,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask,
            )

            optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            optimizer.step()
            losses += loss.item()

        return losses / len(train_dataloader)

    def evaluate(self, model):
        loss_fn = self.loss_fn
        model.eval()
        losses = 0

        val_iter = self.get_data_iterator("valid")
        val_dataloader = DataLoader(
            val_iter, batch_size=BATCH_SIZE, collate_fn=self.collate_fn
        )

        for src, tgt in val_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src, tgt_input
            )

            logits = model(
                src,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask,
            )

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

        return losses / len(val_dataloader)

    def translate(self, model: torch.nn.Module, src_sentence: str):
        model.eval()
        src = self.text_transform[self.SRC_LANGUAGE](src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = greedy_decode(
            model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX
        ).flatten()
        return (
            " ".join(
                self.vocab_transform[self.TGT_LANGUAGE].lookup_tokens(
                    list(tgt_tokens.cpu().numpy())
                )
            )
            .replace("<bos>", "")
            .replace("<eos>", "")
        )

    def generate(self, src):
        src = torch.tensor(src)
        self.transformer.eval()
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = greedy_decode(
            self.transformer,
            src,
            src_mask,
            # max_len=num_tokens + 5,
            max_len=50,
            start_symbol=BOS_IDX,
        ).flatten()
        self.transformer.train(True)
        return list(
            self.vocab_transform[self.TGT_LANGUAGE].lookup_tokens(
                list(tgt_tokens.cpu().numpy())
            )
        )


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        classical_mode=True,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        if classical_mode:
            self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        else:
            self.src_tok_emb = None
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.classical_mode = classical_mode

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        if self.classical_mode:
            src_emb = self.positional_encoding(self.src_tok_emb(src))
        else:
            src_emb = src
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        if not self.classical_mode:
            src_padding_mask = torch.any(src_padding_mask, dim=-1)
            memory_key_padding_mask = torch.any(memory_key_padding_mask, dim=-1)
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        if self.classical_mode:
            return self.transformer.encoder(
                self.positional_encoding(self.src_tok_emb(src)), src_mask
            )
        else:
            return self.transformer.encoder(src, src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt):
    # target mask, padding mask
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


from torch.nn.utils.rnn import pad_sequence


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def tensor_transform(token_ids: List[int]):
    return torch.cat(
        (torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX]))
    )


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(
            DEVICE
        )
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


if __name__ == "__main__":
    t = TranslationTransformer()
    # t.train()
    print(t.translate(t.transformer, "Eine Gruppe von Menschen steht vor einem Iglu ."))
