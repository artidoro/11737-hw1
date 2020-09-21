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
            for item in os.listdir():
                if item.endswith('.model') or item.endswith('.vocab'):
                    print(f'Removing {item}')
                    os.remove(item)
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
################################################################
# MULTILINGUAL
################################################################
import os
import json
from collections import defaultdict
LANGS = ["en", "cs", "es", "ar", "hy", "lt", "af", "ta"]

langs_lines = {}
langs_infos = defaultdict(dict)

for lang in LANGS:
    with open(os.path.join('data', lang, f'{lang}-ud-train.notags')) as infile:
        langs_lines[lang] = [line.strip() for line in infile]
        langs_infos['nb_sent'][lang] = len(langs_lines[lang])
N = sum([langs_infos['nb_sent'][lang] for lang in LANGS])
for lang in LANGS:
    langs_infos['p'][lang] = langs_infos['nb_sent'][lang] / N
for alpha in [0.5, 0.7]:
    P = sum([langs_infos['p'][lang]**alpha for lang in LANGS])
    for lang in LANGS:
        langs_infos[f'q-{alpha}'][lang] = langs_infos['p'][lang]**alpha/P
with open('lang_infos.json', 'w') as out_file:
    out_file.write(json.dumps(langs_infos, sort_keys=True, indent=4))

# %%
ml_lines = []
max_nb_tokens = langs_infos['nb_sent']['cs']
for lang in LANGS:
    nb_lang = langs_infos['q-0.5'][lang] * max_nb_tokens / langs_infos['q-0.5']['cs']
    lang_lines = langs_lines[lang] * int(nb_lang / len(langs_lines[lang]))
    ml_lines += lang_lines

with open(os.path.join('data', 'ml', 'ml-ud-train.notags'), 'w') as out_file:
    for line in ml_lines:
        out_file.write(line + '\n')
#%%
import sentencepiece as spm
from math import floor

ext = ['sm', 'md', 'lg']
path = os.getcwd()

os.chdir(os.path.join(path, 'data', 'ml'))
with open('ml-ud-train.notags') as infile:
    l = len(set(infile.readlines()))
print(f'vocab-size ml: {l}')
#%%
div = 6
while True:
    try:
        p = 1000
        factor = int(l/div)
        sizes = [floor(factor/p)*p, floor(factor/p)*p*2, floor(factor/p)*p*3]
        for j, vocab_size in enumerate(sizes):
            print(sizes)
            spm.SentencePieceTrainer.train(input='ml-ud-train.notags', model_prefix=f'ml-{ext[j]}-{vocab_size}', model_type='bpe', vocab_size=vocab_size)
    except Exception as e:
        print(e)
        div += 1
        for item in os.listdir():
            if item.endswith('.model') or item.endswith('.vocab'):
                print(f'Removing {item}')
                os.remove(item)
        print(f'Trying with div: {div}, factor: {int(l/div)}')
    else:
        break
# os.chdir(path)

# %%
import os
import sentencepiece as spm
from tqdm import tqdm


langs = [elt  for elt in os.listdir('data') if elt != 'ml']
ext = ['sm', 'md', 'lg']
inside_word_token = 'INSIDE_WORD'
path_ml = os.path.join('data', 'ml')

for size in ['sm', 'md']: #, 'lg']:
    prefixed = [filename for filename in os.listdir(path_ml) if filename.startswith(f'ml-{size}') and filename.endswith('model')]
    assert len(prefixed) == 1, prefixed
    sp = spm.SentencePieceProcessor(model_file=os.path.join(path_ml, prefixed[0]))
    for lang in langs:
        print(lang)
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
                    out_file_name = 'ml-' + in_file_name + f'.bpe-{size}'
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
