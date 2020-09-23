import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time
from device import device


class BiLSTMPOSTagger(nn.Module):
    def __init__(
        self,
        dropout=0.5,
        pad_idx=0,
        # Main LSTM
        hidden_dim=100,
        output_dim=10,
        n_layers=2,
        bidirectional=True,
        # Multilingual params
        nb_langs=None,
        lang_embeddding_dim=0,
        # BPE
        bpe_input_dim=0,
        bpe_embedding_dim=0,
        bpe_hidden_dim=0,
        bpe_n_layers=0,
        bpe_bidirectional=0,
        # Char RNN
        char_rnn_input_dim=0,
        char_rnn_embedding_dim=0,
        char_rnn_hidden_dim=0,
        char_rnn_n_layers=0,
        char_rnn_bidirectional=0,
        **kwargs
        # TODO: mBERT params
        # TODO: CRF
    ):
        super().__init__()
        # print('dropout', dropout,'pad_idx', pad_idx,'hidden_dim', hidden_dim,'output_dim', output_dim,'n_layers', n_layers,'bidirectional', bidirectional,'nb_langs', nb_langs,'lang_embeddding_dim', lang_embeddding_dim,'bpe_input_dim', bpe_input_dim,'bpe_embedding_dim', bpe_embedding_dim,'bpe_hidden_dim', bpe_hidden_dim,'bpe_n_layers', bpe_n_layers,'bpe_bidirectional', bpe_bidirectional,'char_rnn_input_dim', char_rnn_input_dim,'char_rnn_embedding_dim', char_rnn_embedding_dim,'char_rnn_hidden_dim', char_rnn_hidden_dim,'char_rnn_n_layers', char_rnn_n_layers,'char_rnn_bidirectiona', char_rnn_bidirectional)
        # ML
        if nb_langs is not None:
            self.lang_embedding = nn.Embedding(nb_langs, lang_embeddding_dim)
        # Char rnn
        if char_rnn_embedding_dim != 0:
            self.using_char_rnn = True
            self.char_rnn_embedding = nn.Embedding(char_rnn_input_dim, char_rnn_embedding_dim)
            self.char_lstm = nn.LSTM(
                input_size=char_rnn_embedding_dim + lang_embeddding_dim,
                hidden_size=char_rnn_hidden_dim,
                num_layers=char_rnn_n_layers,
                bidirectional=char_rnn_bidirectional,
                dropout=dropout if char_rnn_n_layers > 1 else 0,
            )
            self.char_rnn_output_dim = char_rnn_hidden_dim * 2 if char_rnn_bidirectional else char_rnn_hidden_dim
        else:
            self.char_rnn_output_dim = 0
        # BPE
        if bpe_embedding_dim != 0:
            self.using_bpe = True
            self.bpe_embedding = nn.Embedding(bpe_input_dim, bpe_embedding_dim)
            self.bpe_lstm = nn.LSTM(
                input_size=bpe_embedding_dim + lang_embeddding_dim,
                hidden_size=bpe_hidden_dim,
                num_layers=bpe_n_layers,
                bidirectional=bpe_bidirectional,
                dropout=dropout if bpe_n_layers > 1 else 0,
            )
            self.bpe_output_dim = bpe_hidden_dim * 2 if bpe_bidirectional else bpe_hidden_dim
        else:
            self.bpe_output_dim = 0
        # TODO: verify padding index.
        # self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        # Main LSTM.
        self.lstm = nn.LSTM(
            input_size=lang_embeddding_dim + self.char_rnn_output_dim + self.bpe_output_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, chars, chars_len, bpes, bpes_len, bpes_mask, lang_idx=None):

        # text = [sent len, batch size]

        embedded = None
        # embedded = self.dropout(self.embedding(text))
        # # embedded = [sent len, batch size, emb dim]
        
        # Char rnn
        # start = time.time()
        if self.char_rnn_output_dim != 0:
            # Needs to only run on words. We need one vector per word.
            # s = time.time()
            char_rnn_embed = self.dropout(self.char_rnn_embedding(chars)) # [word len, words in batch, embedding size]
            # print(char_rnn_embed.shape)
            if lang_idx is not None:
                batch_lang_idx = torch.tensor((), dtype=torch.long, device=device).new_full(chars.shape, lang_idx, requires_grad=False)
                lang_embedded = self.dropout(self.lang_embedding(batch_lang_idx))
                char_rnn_embed = torch.cat([char_rnn_embed, lang_embedded], dim=2)
            # print('char embed', time.time()-s)
            # s = time.time()
            # print(char_rnn_embed.shape)
            # char_rnn_embed_packed = pack_padded_sequence(char_rnn_embed, chars_len, enforce_sorted=False)
            char_rnn_outputs, _ = self.char_lstm(char_rnn_embed) # [word len, words in batch, outputdim]
            # char_rnn_outputs, _ = pad_packed_sequence(char_rnn_outputs)
            # print('char lstm', time.time()-s)
            # s = time.time()
            # char_rnn_outputs, _ = torch.max(char_rnn_outputs, dim=0) # [words in batch, outputdim]
            char_rnn_outputs = char_rnn_outputs[0] # [words in batch, outputdim] (take the first vector)
            # print('char select', time.time()-s)
            # s = time.time()
            char_rnn_outputs = char_rnn_outputs.view(text.shape[0], text.shape[1], -1) # [sent_len, batch_size, outputdim]
            # print('char view', time.time()-s)
            # s = time.time()
            embedded = torch.cat([embedded, char_rnn_outputs], dim=2) if embedded is not None else char_rnn_outputs
            # print('char cat', time.time()-s)
            # s = time.time()
        # print('chars:', time.time()-start)
        # BPE rnn
        # start = time.time()
        if self.bpe_output_dim != 0:
            bpe_embed = self.dropout(self.bpe_embedding(bpes))
            if lang_idx is not None:
                batch_lang_idx = torch.tensor((), dtype=torch.long, device=device).new_full(bpes.shape, lang_idx, requires_grad=False)
                lang_embedded = self.dropout(self.lang_embedding(batch_lang_idx))
                bpe_embed = torch.cat([bpe_embed, lang_embedded], dim=2)
            bpe_outputs, _ = self.bpe_lstm(bpe_embed)
            bpe_mask = bpes_mask.reshape(-1)
            bpe_outputs = bpe_outputs.view(-1, bpe_outputs.shape[-1])[bpe_mask]
            bpe_outputs = bpe_outputs.view(text.shape[0], text.shape[1], -1)
            embedded = torch.cat([embedded, bpe_outputs], dim=2) if embedded is not None else bpe_outputs
        # print('bpe:', time.time()-start)

        # Lang embed.
        # start = time.time()
        if lang_idx is not None:
            batch_lang_idx = torch.tensor((), dtype=torch.long, device=device).new_full(text.shape, lang_idx, requires_grad=False)
            lang_embedded = self.dropout(self.lang_embedding(batch_lang_idx))
            embedded = torch.cat([embedded, lang_embedded], dim=2)
        # print('lang_idx:', time.time()-start)

        # pass embeddings into LSTM
        # start = time.time()
        outputs, (hidden, cell) = self.lstm(embedded)

        # outputs holds the backward and forward hidden states in the final layer
        # hidden and cell are the backward and forward hidden and cell states at the final time-step

        # output = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]

        # we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(outputs))

        # predictions = [sent len, batch size, output dim]
        # print('main lstm:', time.time()-start)

        return predictions