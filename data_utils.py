from torch.utils.data import IterableDataset
from torch.distributions.categorical import Categorical
import torch

import sentencepiece as spm
from bpemb import BPEmb
from device import device
import os

import json
import time

class POSIteratorML:
    def __init__(self, generators, batch_size, bpe_model='ml-100000', langs=None, nb_samples=None, q_probs=None, pad_idx_char_rnn=412, pad_idx_bpe=100001):
        super(POSIteratorML).__init__()
        self.generators = generators
        self.langs = langs
        self.nb_samples = nb_samples if langs is not None else len(generators) * batch_size
        self.batch_size = batch_size
        self.init_epoch()
        if self.langs is not None:
            self.c = Categorical(torch.tensor([q_probs[lang] for lang in langs]).cuda())
            self.langs = langs

        
        # BPE
        if 'mlours' in bpe_model:
            prefixed = [filename for filename in os.listdir(os.path.join('data', 'ml')) if filename.startswith(f'ml-{bpe_model.split("-")[1]}') and filename.endswith('model')]
            assert len(prefixed) == 1, prefixed
            self.bpe = spm.SentencePieceProcessor(model_file=os.path.join('data', 'ml', prefixed[0]))
            self.bpe_encode = lambda x: torch.tensor(self.bpe.encode(x, out_type=int), dtype=torch.long, device=device, requires_grad=False)
        else:
            vocab_size = int(bpe_model.split('-')[1])
            self.bpe = BPEmb(lang="multi", vs=vocab_size, dim=300)

            def bpe_encode_function(sentences):
                # batchsize x words in sent
                enc_sents = []
                for sentence in sentences:
                    # bpe.encode_ids([word1, word2]) -> [[sbw1-1, sbw1-2], [sbw2-1,...]]
                    enc_sents.append(self.bpe.encode_ids(sentence))
                bpe_mask = []
                enc_sents_flat =[]
                for enc_sent in enc_sents:
                    l = [[True] + [False] * (len(sublist)-1) for sublist in enc_sent]
                    bpe_mask.append([item for sublist in l for item in sublist])
                    assert sum(bpe_mask[-1])==len(enc_sent), str((sum(bpe_mask[-1]), len(enc_sent))) # TODO debug
                    enc_sents_flat.append([item for sublist in enc_sent for item in sublist])
                lens = []
                for sent in enc_sents_flat:
                    lens.append(len(sent))
                max_len = max(lens)
                padded_sents = []
                padded_bpe_mask = []
                for sent, mask in zip(enc_sents_flat, bpe_mask):
                    assert len(sent) == len(mask), str((len(sent), len(mask))) # TODO: remove
                    padded_sents.append(sent + [pad_idx_bpe] * (max_len - len(sent)))
                    padded_bpe_mask.append(mask + [False] * (max_len - len(mask)))
                bpe = torch.tensor(padded_sents, dtype=torch.long, device=device, requires_grad=False)
                bpe_mask = torch.tensor(padded_bpe_mask, dtype=torch.bool, device=device, requires_grad=False)
                bpe_len = torch.tensor(lens, dtype=torch.long, device=device, requires_grad=False)
                return bpe, bpe_len, bpe_mask
            self.bpe_encode = bpe_encode_function
        
        # Char ngram
        with open('char_vocab.json') as infile:
            self.chartoi = json.loads(infile.read())
        def char_encode_function(sentences):
            word_lens = []
            for sentence in sentences:
                for word in sentence:
                    word_lens.append(len(word))
            max_len = max(word_lens)
            enc_sents = []                    
            for sentence in sentences:
                enc_sent = []
                for word in sentence:
                    word_enc = []
                    for c in word:
                        word_enc.append(self.chartoi[c])
                    enc_sent.append(word_enc + [pad_idx_char_rnn] * (max_len - len(word_enc)))
                enc_sents.append(enc_sent)

            chars = torch.tensor(enc_sents, dtype=torch.long, device=device, requires_grad=False).view(-1, max_len)
            chars_len = torch.tensor(word_lens, dtype=torch.long, device=device, requires_grad=False)
            return chars, chars_len
            # torch.tensor([[[self.chartoi[c] for c in word] for word in sentence] for sentence in sentences], dtype=torch.long, device=device)
        self.char_encode = char_encode_function
        
        # mBert
        # self.mBert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    def init_epoch(self):
        if self.langs is not None:
            for lang in self.langs:
                self.generators[lang].init_epoch()
            self.iterator = {lang:self.generators[lang].__iter__() for lang in self.langs}
        else:
            self.iterator = self.generators.__iter__()
        self.count = 0

    def itos_batch(self, batch):
        batch_list = batch.text.transpose(0,1).tolist()
        TEXT = batch.dataset.fields['text']
        return [[TEXT.vocab.itos[i] for i in sent] for sent in batch_list]

    def __iter__(self):
        return self

    def __len__(self):
        return int(self.nb_samples/self.batch_size)
    
    def __next__(self):
        # start = time.time()
        # Stop if finished.
        if self.count >= self.nb_samples:
            raise StopIteration
        self.count += self.batch_size

        lang_idx = None
        if self.langs is not None:
            # Pick a language.
            lang_idx = self.c.sample().item()
            lang = self.langs[lang_idx]

            # Get the text from the batch.
            batch = next(self.iterator[lang])
        else:
            batch = next(self.iterator)
        # string_batch = self.itos_batch(batch) # Includes transpose
        # chars, chars_len = self.char_encode(string_batch)
        # bpes, bpe_len, bpes_mask = self.bpe_encode(string_batch)
        # chars, bpes, bpes_mask = chars.transpose(0,1), bpes.transpose(0,1), bpes_mask.transpose(0,1)
        # print('next:', time.time()-start)
        chars, chars_len, bpes, bpe_len, bpes_mask = None, None, None, None, None         
        return chars, chars_len, bpes, bpe_len, bpes_mask, batch.text, batch.udtags, lang_idx

# class POSIterator:
#     def __init__(self, iterator, batch_size, bpe_model='ml-100000', pad_idx_char_rnn=412, pad_idx_bpe=100001):
#         super(POSIterator).__init__()
#         self.batch_size = batch_size
#         self.iterator = iterator
#         self.nb_samples = len(iterator)
        
#         # BPE
#         if 'mlours' in bpe_model:
#             prefixed = [filename for filename in os.listdir(os.path.join('data', 'ml')) if filename.startswith(f'ml-{bpe_model.split("-")[1]}') and filename.endswith('model')]
#             assert len(prefixed) == 1, prefixed
#             self.bpe = spm.SentencePieceProcessor(model_file=os.path.join('data', 'ml', prefixed[0]))
#             self.bpe_encode = lambda x: torch.tensor(self.bpe.encode(x, out_type=int), dtype=torch.long, device=device, requires_grad=False)
#         else:
#             vocab_size = int(bpe_model.split('-')[1])
#             self.bpe = BPEmb(lang="multi", vs=vocab_size, dim=300)

#             def bpe_encode_function(sentences):
#                 # batchsize x words in sent
#                 enc_sents = []
#                 for sentence in sentences:
#                     # bpe.encode_ids([word1, word2]) -> [[sbw1-1, sbw1-2], [sbw2-1,...]]
#                     enc_sents.append(self.bpe.encode_ids(sentence))
#                 bpe_mask = []
#                 enc_sents_flat =[]
#                 for enc_sent in enc_sents:
#                     l = [[True] + [False] * (len(sublist)-1) for sublist in enc_sent]
#                     bpe_mask.append([item for sublist in l for item in sublist])
#                     assert sum(bpe_mask[-1])==len(enc_sent), str((sum(bpe_mask[-1]), len(enc_sent))) # TODO debug
#                     enc_sents_flat.append([item for sublist in enc_sent for item in sublist])
#                 lens = []
#                 for sent in enc_sents_flat:
#                     lens.append(len(sent))
#                 max_len = max(lens)
#                 padded_sents = []
#                 padded_bpe_mask = []
#                 for sent, mask in zip(enc_sents_flat, bpe_mask):
#                     assert len(sent) == len(mask), str((len(sent), len(mask))) # TODO: remove
#                     padded_sents.append(sent + [pad_idx_bpe] * (max_len - len(sent)))
#                     padded_bpe_mask.append(mask + [False] * (max_len - len(mask)))
#                 bpe = torch.tensor(padded_sents, dtype=torch.long, device=device, requires_grad=False)
#                 bpe_mask = torch.tensor(padded_bpe_mask, dtype=torch.bool, device=device, requires_grad=False)
#                 bpe_len = torch.tensor(lens, dtype=torch.long, device=device, requires_grad=False)
#                 return bpe, bpe_len, bpe_mask
#             self.bpe_encode = bpe_encode_function
        
#         # Char ngram
#         with open('char_vocab.json') as infile:
#             self.chartoi = json.loads(infile.read())
#         def char_encode_function(sentences):
#             word_lens = []
#             for sentence in sentences:
#                 for word in sentence:
#                     word_lens.append(len(word))
#             max_len = max(word_lens)
#             enc_sents = []                    
#             for sentence in sentences:
#                 enc_sent = []
#                 for word in sentence:
#                     word_enc = []
#                     for c in word:
#                         word_enc.append(self.chartoi[c])
#                     enc_sent.append(word_enc + [pad_idx_char_rnn] * (max_len - len(word_enc)))
#                 enc_sents.append(enc_sent)

#             chars = torch.tensor(enc_sents, dtype=torch.long, device=device, requires_grad=False).view(-1, max_len)
#             chars_len = torch.tensor(word_lens, dtype=torch.long, device=device, requires_grad=False)
#             return chars, chars_len
#             # torch.tensor([[[self.chartoi[c] for c in word] for word in sentence] for sentence in sentences], dtype=torch.long, device=device)
#         self.char_encode = char_encode_function
        
#         # mBert
#         # self.mBert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

#     def itos_batch(self, batch):
#         batch_list = batch.text.transpose(0,1).tolist()
#         TEXT = batch.dataset.fields['text']
#         return [[TEXT.vocab.itos[i] for i in sent] for sent in batch_list]
#     def init_epoch(self):
#         self.count = 0

#     def __iter__(self):
#         return self

#     def __len__(self):
#         return int(self.nb_samples/self.batch_size)
    
#     def __next__(self):
#         # start = time.time()
#         # Stop if finished.
#         if self.count >= self.nb_samples:
#             raise StopIteration
#         self.count += self.batch_size

#         # Get the text from the batch.
#         batch = next(self.iterator)
#         string_batch = self.itos_batch(batch) # Includes transpose
#         chars, chars_len = self.char_encode(string_batch)
#         bpes, bpe_len, bpes_mask = self.bpe_encode(string_batch)
#         chars, bpes, bpes_mask = chars.transpose(0,1), bpes.transpose(0,1), bpes_mask.transpose(0,1)
#         # print('next:', time.time()-start)
                        
#         return chars, chars_len, bpes, bpe_len, bpes_mask, batch.text, batch.udtags

# class POSIterator:
#     def __init__(self, iterator, bpe_model='ml-100000'):
#         super(POSIterator).__init__()
#         self.iterator = iterator
#         nb_samples = len(iterator)
        
#         # BPE
#         if 'mlours' in bpe_model:
#             prefixed = [filename for filename in os.listdir(path_ml) if filename.startswith(f'ml-{size}') and filename.endswith('model')]
#             assert len(prefixed) == 1, prefixed
#             self.bpe = spm.SentencePieceProcessor(model_file=os.path.join('data', 'ml', prefix[0]))
#             self.bpe_encode = lambda x: torch.tensor(self.bpe.encode(x, out_type=int), dtype=long, device=device)
#         else:
#             vocab_size = int(bpe_model.split('-')[1])
#             self.bpe = BPEmb(lang="multi", vs=vocab_size, dim=300)
#             self.bpe_encode = lambda x: torch.tensor(self.bpe.encode_ids(x), dtype=torch.long, device=device)
        
#         # Char ngram
#         with open('char_vocab.json') as infile:
#             self.chartoi = json.loads(infile.read())
#         self.char_encode = lambda sentences: torch.tensor([[[self.chartoi[c] for c in word] for word in sentence] for sentence in sentences], dtype=torch.long, device=device)
        
#         # mBert
#         # self.mBert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

#     def itos_batch(self, batch):
#         batch_list = batch.tolist()
#         TEXT = batch.dataset.fields['text']
#         return [[TEXT.vocab.itos[i] for i in sent] for sent in batch]

#     def __iter__(self):
#         return self
    
#     def __next__(self):
#         # Stop if finished.
#         if self.c >= self.nb_samples:
#             raise StopIteration

#         # Get the text from the batch.
#         batch = next(self.iterator)
#         string_batch = self.itos_batch(batch)

#         chars = self.char_encode(string_batch)
#         bpes = self.bpe_encode(string_batch)
#         bpes_mask = torch.tensor([[word.startswith('_') for word in sentence] for sentence in string_batch], dtype=torch.bool, device=device)
                        
#         return chars, bpes, bpes_mask, batch.text, batch.udtags, lang_idx

    


