#%%
import os
for lang in os.listdir('data'):
    with open(os.path.join('data', lang, f'{lang}-ud-train.conll')) as infile,\
        open(os.path.join('data', lang, f'{lang}-ud-train.notags'), 'w') as outfile:
        for line in infile:
            outfile.write(line.split('\t')[0].lower()+'\n')
# %%
import os
import sentencepiece as spm
from tqdm import tqdm
from math import floor

langs = os.listdir('data')
ext = ['sm', 'md', 'lg']
path = os.getcwd()
for lang in langs:
    os.chdir(os.path.join(path, 'data', lang))
    with open(f'{lang}-ud-train.notags') as infile:
        l = len(set(infile.readlines()))
    print(f'vocab-size {lang}: {l}')
    div = 3
    while True:
        try:
            p = 100
            factor = int(l/div)
            sizes = [floor(factor/p)*p, floor(factor/p)*p*2, floor(factor/p)*p*3]
            for j, vocab_size in enumerate(sizes):
                print(sizes)
                spm.SentencePieceTrainer.train(input=f'{lang}-ud-train.notags', model_prefix=f'{lang}-{ext[j]}-{vocab_size}', model_type='bpe', vocab_size=vocab_size)
        except Exception as e:
            print(e)
            div += 1
            print(f'Trying with div: {div}, factor: {int(l/div)}')
        else:
            break

os.chdir(path)


# %%
import os
import sentencepiece as spm
from tqdm import tqdm


langs = os.listdir('data')
ext = ['sm', 'md', 'lg']
inside_word_token = 'INSIDE_WORD'

for lang in langs:
    path=os.path.join('data', lang)
    for in_file_name in [f'{lang}-ud-{mode}.conll' for mode in ['test', 'dev', 'train']]:
        with open(os.path.join(path, in_file_name)) as in_file:
            words, tags = [], []
            for line in in_file.readlines():
                split = line.strip().split('\t')
                if len(split) > 1:
                    word, tag = split[0], split[1]
                    words.append(word)
                    tags.append(tag)
                else:
                    words.append('')
                    tags.append('')

            for size in ['sm', 'md', 'lg']:
                prefixed = [filename for filename in os.listdir(path) if filename.startswith(f'{lang}-{size}') and filename.endswith('model')]
                assert len(prefixed) == 1, prefixed
                sp = spm.SentencePieceProcessor(model_file=os.path.join(path, prefixed[0]))
                out_file_name = in_file_name + f'.bpe-{size}'
                with open(os.path.join(path, out_file_name), 'w') as out_file:
                    encoded_words = sp.encode(words, out_type=str)
                    assert len(encoded_words) == len(tags)
                    for encoded_word, tag in zip(encoded_words, tags):
                        if tag == '':
                            out_file.write('\n')
                        else:
                            for i, token in enumerate(encoded_word):
                                if i == 0:
                                    out_file.write(f'{token}\t{tag}\n')
                                else:
                                    out_file.write(f'{token}\t{inside_word_token}\n')

# %%
