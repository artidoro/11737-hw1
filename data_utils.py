from torch.utils.data import IterableDataset
from torch.distributions.categorical import Categorical
import torch

import sentencepiece as spm
from bpemb import BPEmb

import json


class POSIteratorML:
    def __init__(self, generators, batch_size, bpe_model='ml-100000', langs=None, nb_samples=None, q_probs=None):
        super(POSIteratorML).__init__()
        self.c = Categorical(torch.tensor([q_probs[lang] for lang in langs]).cuda())
        self.generators = generators
        self.langs = langs
        self.nb_samples = nb_samples
        self.batch_size = batch_size
        self.init_epoch()
        
        # BPE
        if 'mlours' in bpe_model:
            prefixed = [filename for filename in os.listdir(path_ml) if filename.startswith(f'ml-{size}') and filename.endswith('model')]
            assert len(prefixed) == 1, prefixed
            self.bpe = spm.SentencePieceProcessor(model_file=os.path.join('data', 'ml', prefix[0]))
            self.bpe_encode = lambda x: torch.tensor(self.bpe.encode(x, out_type=int), dtype=long, device='cuda', requires_grad=False)
        else:
            vocab_size = int(bpe_model.split('-')[1])
            self.bpe = BPEmb(lang="multi", vs=vocab_size, dim=300)
            self.bpe_encode = lambda x: torch.tensor(self.bpe.encode_ids(x), dtype=torch.long, device='cuda', requires_grad=False)
        
        # Char ngram
        with open('char_vocab.json') as infile:
            self.chartoi = json.loads(infile.read())
        def char_encode_function(sentences):
            enc_sents = []
            for sentence in sentences:
                enc_sent = []
                for word in sentence:
                    word_enc = []
                    for c in word:
                        word_enc.append(self.chartoi[c])
                    enc_sent.append(word_enc)
                enc_sents.append(enc_sent)
            return torch.tensor(enc_sents, dtype=torch.long, device='cuda', requires_grad=False)

            # torch.tensor([[[self.chartoi[c] for c in word] for word in sentence] for sentence in sentences], dtype=torch.long, device='cuda')
        self.char_encode = char_encode_function
        
        # mBert
        # self.mBert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    def init_epoch(self):
        for lang in self.langs:
            self.generators[lang].init_epoch()
        self.iterator = {lang:self.generators[lang].__iter__() for lang in self.langs}
        self.count = 0

    def itos_batch(self, batch):
        batch_list = batch.text.tolist()
        TEXT = batch.dataset.fields['text']
        return [[TEXT.vocab.itos[i] for i in sent] for sent in batch_list]

    def __iter__(self):
        return self
    
    def __next__(self):
        # Stop if finished.
        if self.count >= self.nb_samples:
            raise StopIteration
        self.count += self.batch_size

        # Pick a language.
        lang_idx = self.c.sample().item()
        lang = self.langs[lang_idx]

        # Get the text from the batch.
        batch = next(self.iterator[lang])
        string_batch = self.itos_batch(batch)

        chars = self.char_encode(string_batch)
        bpes = self.bpe_encode(string_batch)
        bpes_mask = torch.tensor([[word.startswith('_') for word in sentence] for sentence in string_batch], dtype=torch.bool, device='cuda')
                        
        return chars, bpes, bpes_mask, batch.text, batch.udtags, lang_idx


class POSIterator:
    def __init__(self, iterator, bpe_model='ml-100000'):
        super(POSIterator).__init__()
        self.iterator = iterator
        nb_samples = len(iterator)
        
        # BPE
        if 'mlours' in bpe_model:
            prefixed = [filename for filename in os.listdir(path_ml) if filename.startswith(f'ml-{size}') and filename.endswith('model')]
            assert len(prefixed) == 1, prefixed
            self.bpe = spm.SentencePieceProcessor(model_file=os.path.join('data', 'ml', prefix[0]))
            self.bpe_encode = lambda x: torch.tensor(self.bpe.encode(x, out_type=int), dtype=long, device='cuda')
        else:
            vocab_size = int(bpe_model.split('-')[1])
            self.bpe = BPEmb(lang="multi", vs=vocab_size, dim=300)
            self.bpe_encode = lambda x: torch.tensor(self.bpe.encode_ids(x), dtype=torch.long, device='cuda')
        
        # Char ngram
        with open('char_vocab.json') as infile:
            self.chartoi = json.loads(infile.read())
        self.char_encode = lambda sentences: torch.tensor([[[self.chartoi[c] for c in word] for word in sentence] for sentence in sentences], dtype=torch.long, device='cuda')
        
        # mBert
        # self.mBert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    def itos_batch(self, batch):
        batch_list = batch.tolist()
        TEXT = batch.dataset.fields['text']
        return [[TEXT.vocab.itos[i] for i in sent] for sent in batch]

    def __iter__(self):
        return self
    
    def __next__(self):
        # Stop if finished.
        if self.c >= self.nb_samples:
            raise StopIteration

        # Get the text from the batch.
        batch = next(self.iterator)
        string_batch = self.itos_batch(batch)

        chars = self.char_encode(string_batch)
        bpes = self.bpe_encode(string_batch)
        bpes_mask = torch.tensor([[word.startswith('_') for word in sentence] for sentence in string_batch], dtype=torch.bool, device='cuda')
                        
        return chars, bpes, bpes_mask, batch.text, batch.udtags, lang_idx

    


