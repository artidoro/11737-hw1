#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
languages = ['en','es','cs','ar','af','lt','hy','ta']
nb_tokens = [204586, 382436, 719317, 225853, 33894, 47605, 42105, 6329]
performance = [91.87, 94.05, 94.41, 94.22, 89.96, 76.07, 81.34, 41.30]

# %%
plt.scatter(nb_tokens, performance)
for i, lang in enumerate(languages):
    plt.text(nb_tokens[i]+10000, performance[i]-1, lang, fontsize=9)
plt.xlabel("Number of tokens in training set")
plt.ylabel("POS tagging test accuracy")
# %%

# %%
