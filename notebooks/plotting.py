import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE


def TSNE_scatterplot(model, word, neg_word, save=False):
    close_words = np.array(model.wv.most_similar([word]))
    neg_words = np.array(model.wv.most_similar(negative=[neg_word]))

    arrays = np.vstack((model.wv[[word]], model.wv[close_words[:, 0]], model.wv[neg_words[:, 0]]))
    word_labels = [word] + list(close_words[:, 0]) + list(neg_words[:, 0])

    color_list = ['blue'] * len(close_words) + ['green'] * len(neg_words)

    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(arrays)

    x = Y[1:, 0]
    y = Y[1:, 1]
    plt.figure(figsize=(11, 4))
    plt.scatter(Y[0, 0], Y[0, 1], marker="*", c='blue', s=120)
    plt.scatter(x, y, c=color_list)
    plt.grid()

    annotations = []

    for line in range(len(word_labels)):
        annotations.append(plt.text(Y[line, 0], Y[line, 1], word_labels[line].title()))

    if save:
        plt.savefig(f"{str(model)}.png", bbox_inches='tight')
