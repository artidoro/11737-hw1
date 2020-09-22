#%%
import os
from collections import defaultdict

min_count = 1
char_vocab = defaultdict(int)
for lang in [lang for lang in os.listdir('data') if lang != 'ml']:
    with open(os.path.join('data', lang, f'{lang}-ud-train.notags')) as infile:
        words = [line.strip() for line in infile.readlines()]
        for word in words:
            for c in word:
                char_vocab[c] += 1

char_vocab = {c:i for i, (c,v) in enumerate(char_vocab.items()) if v >= min_count}
print(len(char_vocab))
#%%
import json
with open('char_ngram_vocab.json', 'w') as out_file:
    out_file.write(json.dumps(char_vocab, indent=4))

# %%
