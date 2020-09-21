import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
# from transformers import AutoTokenizer, AutoModelWithLMHead
# from bpemb import BPEmb
import sentencepiece as spm

from torchtext import data
from torchtext import datasets

from model import BiLSTMPOSTagger

import numpy as np
from collections import Counter
from tqdm import tqdm
import time
import random
import os, sys
import argparse
import json

import warnings

warnings.filterwarnings("ignore")

INSIDE_WORD = 'INSIDE_WORD'
LANGS = ["en", "cs", "es", "ar", "hy", "lt", "af", "ta"]
# set command line options
parser = argparse.ArgumentParser(description="main.py")
parser.add_argument(
    "--mode",
    type=str,
    choices=["train", "eval"],
    default="train",
    help="Run mode",
)
parser.add_argument(
    "--model-name",
    type=str,
    default=None,
    help="name of the saved model",
)
parser.add_argument(
    "--noise",
    type=float,
    default=0.0,
    help="add noise to labels during training"
)
parser.add_argument(
    "--token",
    type=str,
    default='',
    help="specifies which subword tokenization methods to use"
)
args = parser.parse_args()

if not os.path.exists("saved_models"):
    os.mkdir("saved_models")

if args.model_name is None:
    args.model_name = "{}-model".format(args.lang)

# set a fixed seed for reproducibility
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

params = json.load(open("config.json"))

def main():
    print("Running main.py in {} mode with lang: {}".format(args.mode, args.lang))
    # define fields for your data, we have two: the text itself and the POS tag
    TEXT = data.Field(lower=True)
    UD_TAGS = data.Field()

    fields = (("text", TEXT), ("udtags", UD_TAGS))

    # load the data from the specific path
    ext = f'.{args.token}' if args.token != '' else ''
    train_data, valid_data, test_data = datasets.UDPOS.splits(
        fields=fields,
        path=os.path.join("data", args.lang),
        train="{}-ud-train.conll{}".format(args.lang, ext),
        validation="{}-ud-dev.conll{}".format(args.lang, ext),
        test="{}-ud-test.conll{}".format(args.lang, ext),
    )  # modify this to include our own dataset
    # building the vocabulary for both text and the labels
    MIN_FREQ = 2

    TEXT.build_vocab(train_data, min_freq=MIN_FREQ)
    UD_TAGS.build_vocab(train_data)

    if args.mode == "train":
        print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
        print(f"Unique tokens in UD_TAG vocabulary: {len(UD_TAGS.vocab)}")
        print()
        print(f"Number of training examples: {len(train_data)}")
        print(f"Number of validation examples: {len(valid_data)}")

        print(f"Number of tokens in the training set: {sum(TEXT.vocab.freqs.values())}")

    print(f"Number of testing examples: {len(test_data)}")

    if args.mode == "train":
        print("Tag\t\tCount\t\tPercentage\n")
        for tag, count, percent in tag_percentage(UD_TAGS.vocab.freqs.most_common()):
            print(f"{tag}\t\t{count}\t\t{percent*100:4.1f}%")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=params["batch_size"],
        device=device,
    )

    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    model = BiLSTMPOSTagger(
        input_dim=len(TEXT.vocab),
        embedding_dim=params["embedding_dim"],
        hidden_dim=params["hidden_dim"],
        output_dim=len(UD_TAGS.vocab),
        n_layers=params["n_layers"],
        bidirectional=params["bidirectional"],
        dropout=params["dropout"],
        pad_idx=PAD_IDX,
    )

    if args.mode == "train":

        def init_weights(m):
            for name, param in m.named_parameters():
                nn.init.normal_(param.data, mean=0, std=0.1)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        model.apply(init_weights)
        print(f"The model has {count_parameters(model):,} trainable parameters")
        model.embedding.weight.data[PAD_IDX] = torch.zeros(params["embedding_dim"])
        optimizer = optim.Adam(model.parameters())

    TAG_PAD_IDX = UD_TAGS.vocab.stoi[UD_TAGS.pad_token]
    TAG_UNK_IDX = UD_TAGS.vocab.unk_index
    INSIDE_WORD_IDX = UD_TAGS.vocab.stoi[INSIDE_WORD]
    criterion = nn.CrossEntropyLoss(
        ignore_index=TAG_PAD_IDX,
        weight=torch.tensor([0.0 if (
            UD_TAGS.vocab.itos[c] == INSIDE_WORD or
            UD_TAGS.vocab.itos[c] == UD_TAGS.pad_token
            ) else 1.0 for c in range(len(UD_TAGS.vocab))
        ])
    )

    model = model.to(device)
    criterion = criterion.to(device)

    if args.mode == "train":
        N_EPOCHS = 10
        best_valid_loss = float("inf")
        for epoch in range(N_EPOCHS):
            start_time = time.time()
            train_loss, train_acc = train(
                model,
                train_iterator,
                optimizer,
                criterion,
                TAG_PAD_IDX,
                TAG_UNK_IDX,
                INSIDE_WORD_IDX,
                UD_TAGS,
                noise=args.noise
            )
            valid_loss, valid_acc = evaluate(
                model, valid_iterator, criterion, TAG_PAD_IDX, TAG_UNK_IDX, INSIDE_WORD_IDX
            )
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(
                    model.state_dict(), "saved_models/{}.pt".format(args.model_name)
                )

            print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
            print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")

    try:
        model.load_state_dict(torch.load("saved_models/{}.pt".format(args.model_name)))
    except Exception as e:
        print(
            "Model file `{}` doesn't exist. You need to train the model by running this code in train mode. Run python main.py --help for more instructions".format(
                "saved_models/{}.pt".format(args.model_name)
            )
        )
        return

    test_loss, test_acc, test_p, test_r, test_f1, counter = evaluate(
        model, test_iterator, criterion, TAG_PAD_IDX, TAG_UNK_IDX, INSIDE_WORD_IDX, UD_TAGS
    )
    print(f"Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%")
    print(f"Number of tokens in the training set: {sum(TEXT.vocab.freqs.values())}")
    print("Tag\t\tCount\t\tPercent\t\tP\t\tR\t  \tF1\n")
    for tag, count, percent in tag_percentage(counter.most_common()):
        print(f"{tag}\t\t{count}\t\t{percent*100:4.1f}%\t\t{test_p[tag]*100:4.1f}%\t\t{test_r[tag]*100:4.1f}%\t\t{test_f1[tag]*100:4.1f}%")

def tag_percentage(tag_counts):
    total_count = sum([count for tag, count in tag_counts])
    tag_counts_percentages = [
        (tag, count, count / total_count) for tag, count in tag_counts
    ]
    return tag_counts_percentages


def categorical_accuracy(preds, y, tag_pad_idx, tag_unk_idx, inside_word_idx, UD_TAGS=None):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(
        dim=1, keepdim=True
    )  # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx) & (y != tag_unk_idx) & (y != inside_word_idx)
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    ret = correct.float().sum(), y[torch.nonzero(non_pad_elements)].shape[0]

    if UD_TAGS != None:
        correct_tag_counter = Counter()
        precision_denomin_counter = Counter()
        recall_denomin_counter = Counter()
        for tag in UD_TAGS.vocab.freqs.keys():
            tag_idx = UD_TAGS.vocab.stoi[tag]
            if tag_idx == tag_pad_idx or tag_idx == tag_unk_idx or tag_idx == inside_word_idx:
                continue
            y_tag_elements = (y == tag_idx) & non_pad_elements
            pred_tag_elements = (max_preds.squeeze(1) == tag_idx) & non_pad_elements
            correct_tag_counter[tag] = max_preds[y_tag_elements].squeeze(1).eq(y[y_tag_elements]).float().sum().item()
            precision_denomin_counter[tag] = torch.sum(pred_tag_elements).float().item()
            recall_denomin_counter[tag] = torch.sum(y_tag_elements).float().item()
        return ret[0], ret[1], correct_tag_counter, precision_denomin_counter, recall_denomin_counter
    else:
        return ret

def train(model, iterator, optimizer, criterion, tag_pad_idx, tag_unk_idx, inside_word_idx, UD_TAGS=None, noise=0):

    epoch_loss = 0
    epoch_correct = 0
    epoch_n_label = 0

    model.train()

    if noise > 0:
        counts = [UD_TAGS.vocab.freqs[UD_TAGS.vocab.itos[k]] if UD_TAGS.vocab.itos[k] in UD_TAGS.vocab.freqs else 0 for k in range(len(UD_TAGS.vocab))]
        c = Categorical(torch.tensor(counts).cuda()/float(sum(UD_TAGS.vocab.freqs.values())))
        b = Bernoulli(probs=torch.tensor([noise]).cuda())


    for batch in tqdm(iterator):

        text = batch.text
        tags = batch.udtags

        optimizer.zero_grad()

        # text = [sent len, batch size]

        predictions = model(text)

        # predictions = [sent len, batch size, output dim]
        # tags = [sent len, batch size]

        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)

        if noise > 0:
            assert noise >= 0 and noise <= 1
            non_pad_elements = tags != tag_pad_idx
            prob_mask = (b.sample(tags.shape) == 1).squeeze(1).cuda() & non_pad_elements
            noisy_preds = c.sample((torch.sum(prob_mask).item(),))
            # noisy_preds = random.choices([UD_TAGS.vocab.stoi[elt] for elt in UD_TAGS.vocab.freqs.keys()], k=torch.sum(prob_mask).item())
            tags[prob_mask] = torch.tensor(noisy_preds)

        # predictions = [sent len * batch size, output dim]
        # tags = [sent len * batch size]

        loss = criterion(predictions, tags)

        correct, n_labels = categorical_accuracy(
            predictions, tags, tag_pad_idx, tag_unk_idx, inside_word_idx
        )

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_correct += correct.item()
        epoch_n_label += n_labels

    return epoch_loss / len(iterator), epoch_correct / epoch_n_label


def evaluate(model, iterator, criterion, tag_pad_idx, tag_unk_idx, inside_word_idx, UD_TAGS=None):

    epoch_loss = 0
    epoch_correct = 0
    epoch_n_label = 0

    model.eval()

    correct_tag_counter = Counter()
    precision_denomin_counter = Counter()
    recall_denomin_counter = Counter()
    with torch.no_grad():

        for batch in iterator:

            text = batch.text
            tags = batch.udtags

            predictions = model(text)

            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)

            loss = criterion(predictions, tags)

            if UD_TAGS != None:
                correct, n_labels, c_counter, p_counter, r_counter = categorical_accuracy(
                    predictions, tags, tag_pad_idx, tag_unk_idx, inside_word_idx, UD_TAGS
                )
                correct_tag_counter += c_counter
                precision_denomin_counter += p_counter
                recall_denomin_counter += r_counter
            else:
                correct, n_labels= categorical_accuracy(
                    predictions, tags, tag_pad_idx, tag_unk_idx, inside_word_idx
                )
            epoch_loss += loss.item()
            epoch_correct += correct.item()
            epoch_n_label += n_labels

    if UD_TAGS != None:
        precision, recall, f1 = dict(), dict(), dict()
        for tag in UD_TAGS.vocab.freqs.keys():
            if tag in precision_denomin_counter:
                precision[tag] = correct_tag_counter[tag]/precision_denomin_counter[tag]
            else:
                precision[tag] =  f1[tag] = -0.01
            if tag in recall_denomin_counter:
                recall[tag] = correct_tag_counter[tag]/recall_denomin_counter[tag]
            else:
                recall[tag] = -0.01
            if tag in precision_denomin_counter and tag in recall_denomin_counter and precision[tag] != 0 and recall[tag] != 0:
                f1[tag] = 2*precision[tag]*recall[tag]/(precision[tag]+recall[tag])
            else:
                f1[tag] = -0.01
        return epoch_loss / len(iterator), epoch_correct / epoch_n_label, precision, recall, f1, recall_denomin_counter
    else:
        return epoch_loss / len(iterator), epoch_correct / epoch_n_label


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    main()
