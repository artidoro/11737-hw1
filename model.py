import torch
import torch.nn as nn


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

        # ML
        if nb_langs is not None:
            self.lang_embedding = nn.Embedding(nb_langs, lang_embeddding_dim)
        # Char rnn
        if char_rnn_embedding_dim != 0:
            self.using_char_rnn = True
            self.char_rnn_embedding = nn.Embedding(char_rnn_input_dim, char_rnn_embedding_dim)
            self.char_lstm = nn.LSTM(
                char_rnn_embedding_dim,
                char_rnn_hidden_dim,
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
                bpe_embedding_dim,
                bpe_hidden_dim,
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
            lang_embeddding_dim + self.char_rnn_output_dim + self.bpe_output_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, chars, bpes, bpes_mask, lang_idx=None):

        # text = [sent len, batch size]

        embedded = torch.tensor((), device='cuda', type=torch.float)
        # embedded = self.dropout(self.embedding(text))
        # # embedded = [sent len, batch size, emb dim]
        
        # Char rnn
        if self.char_rnn_output_dim != 0:
            # Needs to only run on words. We need one vector per word.
            char_rnn_embed = self.dropout(self.char_rnn_embed(chars))
            char_rnn_outputs, _ = self.char_lstm(char_rnn_embed)
            embedded = torch.cat([embedded, char_rnn_outputs], dim=2)

        # BPE rnn
        if self.bpe_output_dim != 0:
            bpe_embed = self.dropout(self.bpe_embed(chars))
            bpe_outputs, _ = self.char_lstm(bpe_embed)
            bpe_outputs = bpe_outputs[bpes_mask]
            embedded = torch.cat([embedded, bpe_outputs], dim=2)

        # Lang embed.
        if lang_idx is not None:
            lang_embedded = self.dropout(self.lang_embedding(lang_idx))
            embedded = torch.cat([embedded, lang_embedded], dim=2)
        
        # pass embeddings into LSTM
        outputs, (hidden, cell) = self.lstm(embedded)

        # outputs holds the backward and forward hidden states in the final layer
        # hidden and cell are the backward and forward hidden and cell states at the final time-step

        # output = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]

        # we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(outputs))

        # predictions = [sent len, batch size, output dim]

        return predictions