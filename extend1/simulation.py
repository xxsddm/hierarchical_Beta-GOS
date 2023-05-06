import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Gibbs import Gibbs

np.random.seed(4399)

n = [200, 175, 200, 150, 225, 250]
niterate = 100
total = 15
ncluster = [min(np.random.poisson(5) + 3, total - 1) for i in range(len(n))]
container = []
mu = [[] for _ in range(len(n))]
unique_mu = set()
data = [[] for _ in range(len(n))]
title = "ai=3, bi=1"

for i in range(total):
    container.append(np.random.normal(i - total / 2, 0.1))

for i in range(len(n)):
    temp = np.random.choice(container, ncluster[i], replace=False)
    for j in temp:
        mu[i].append(j)
        unique_mu.add(j)

for i in range(len(n)):
    length = 0
    while length < n[i]:
        subLen = 0
        while subLen == 0:
            subLen = min(np.random.poisson(np.random.exponential(4)), n[i] - length)
        length += subLen
        idx = np.random.choice([_ for _ in range(ncluster[i])])
        data[i].append(np.random.normal(mu[i][idx], 0.2, subLen))

data = [np.concatenate(data[i]) for i in range(len(n))]

an = [np.array([3 for _ in range(n[i])]) for i in range(len(n))]
bn = [np.array([1 for _ in range(n[i])]) for i in range(len(n))]
c = [np.array([j for j in range(len(data[i]))], dtype=np.int32) for i in range(len(n))]

summarydata = np.concatenate(data)
stdvar = np.std(summarydata)
mean = np.mean(summarydata)
for i in range(len(n)):
    data[i] = (data[i] - mean) / stdvar

mu_hat, idx2cluster, prob = Gibbs(data=data, an=an, bn=bn, c=c,
                                  dp=0.005, niterate=niterate, draw=True,
                                  title=title)

mu_hat = mu_hat * stdvar + mean
stat_counter = [0] * len(mu_hat)
stat_mu = np.zeros(len(mu_hat), dtype=np.float64)

memo, idx = dict(), 1
for i in range(len(n)):
    for j in range(len(data[i])):
        if idx2cluster[i][j] not in memo:
            memo[idx2cluster[i][j]] = idx
            idx2cluster[i][j] = idx
            idx += 1
        else:
            idx2cluster[i][j] = memo[idx2cluster[i][j]]

for i in range(len(n)):
    data[i] = data[i] * stdvar + mean
    for j in range(len(data[i])):
        stat_counter[idx2cluster[i][j] - 1] += 1
        stat_mu[idx2cluster[i][j] - 1] += data[i][j]

for i in range(len(mu_hat)):
    stat_mu[i] /= stat_counter[i]

print("实际均值: ", sorted([float("{:.4f}".format(i)) for i in list(unique_mu)]))
print("估计均值: ", sorted([float("{:.4f}".format(i)) for i in stat_mu]))
# print("估计均值: ", sorted([float("{:.4f}".format(i)) for i in mu_hat]))

summaryidx = []
summaryprob = []
dataIdx = []
clusterIdx = []
for i in range(len(n)):
    summaryidx.extend([i + 1 for i in range(len(data[i]))])
    summaryprob.extend(prob[i])
    dataIdx.extend([i] * len(data[i]))
    clusterIdx.extend(idx2cluster[i])

plt.figure(figsize=(10, 8))
sns.scatterplot(x=summaryidx, y=summarydata,
                style=dataIdx, hue=clusterIdx,
                legend=False,
                palette=sns.color_palette("colorblind", len(mu_hat)))
plt.xlabel("样本索引")
plt.ylabel("样本值")
plt.savefig()
plt.show()
