import numpy as np
import matplotlib.pyplot as plt


def auc(fpr, tpr, auc, ks, title, save_as):
    fig = plt.figure(figsize=(8, 6))
    plt.title('Receiver Operating Characteristic - %s' % title)
    plt.plot(fpr, tpr, 'b',
             label='AUC=%0.2f, KS=%0.2f' % (auc, ks),
             linewidth=2)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    path = '%s_auc.jpg' % save_as
    plt.savefig(path)
    plt.close()
    return path


def ks(fpr, tpr, ks, save_as):
    plt.figure(figsize=(8, 6))
    plt.title("Kolmogorov Smirnov - %0.2f" % ks)
    plt.plot(tpr, linewidth=2)
    plt.plot(fpr, linewidth=2)
    path = '%s_ks.jpg' % save_as
    plt.savefig(path)
    plt.close()
    return path


def score_dist(score, save_as):
    num = len(score)
    n = 0.04
    cutls = np.arange(0, 1 + n, n)
    score_ls = [0 for i in range(len(cutls) - 1)]
    for s in score:
        pos = int(s * 100 / (n * 100))
        score_ls[pos] += 1
    score_ls = [x / num for x in score_ls]
    x = range(len(score_ls))
    plt.figure(figsize=(8, 6))
    plt.title("Score Distribution")
    plt.bar(x, score_ls, tick_label=cutls[:-1])
    plt.ylim(0, max(score_ls) + 0.05)
    plt.xticks(rotation=45)
    path = '%s_dist.jpg' % save_as
    plt.savefig(path)
    plt.close()
    return path


def feature_dist(tit, data):
    size = len(data)
    label_list = [_[0] for _ in data]
    a = [_[1][0] for _ in data]
    b = [_[1][1] for _ in data]
    x = np.arange(size)

    total_width, n = 0.8, 2     # 有多少个类型，只需更改n即可
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.bar(x, a,  width=width, label='train')
    plt.bar(x + width, b, width=width, label='verify')
    plt.xticks([index + 0.2 for index in x], label_list, rotation=-20)
    plt.title(tit)
    plt.legend()
    plt.show()
