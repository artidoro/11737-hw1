import torch
import torch.nn as nn


class BiLSTMPOSTagger(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
        pad_idx,
        nb_langs,
        lang_embeddding_dim
    ):

        super().__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.lang_embedding = nn.Embedding(nb_langs, lang_embeddding_dim)
        self.lstm = nn.LSTM(
            embedding_dim + lang_embeddding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, lang_idx):

        # text = [sent len, batch size]

        # pass text through embedding layer
        embedded = self.dropout(self.embedding(text))
        lang_embedded = self.dropout(self.lang_embedding(lang_idx))

        # embedded = [sent len, batch size, emb dim]
        print(embedded.shape)
        concat = torch.cat([embedded, lang_embedded], dim=2)
        print(concat.shape)
        # pass embeddings into LSTM
        outputs, (hidden, cell) = self.lstm(concat)

        # outputs holds the backward and forward hidden states in the final layer
        # hidden and cell are the backward and forward hidden and cell states at the final time-step

        # output = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]

        # we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(outputs))

        # predictions = [sent len, batch size, output dim]

        return predictions