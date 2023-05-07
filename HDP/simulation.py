import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Gibbs import Gibbs

np.random.seed(4399)

n = [200, 175, 200, 150, 225, 250]
niterate = 10000
total = 15
ncluster = [min(np.random.poisson(5) + 3, total - 1) for i in range(len(n))]
mu_container = []
std_container = []
mu = [[] for _ in range(len(n))]
std = [[] for _ in range(len(n))]
uniqueIdx = set()
data = [[] for _ in range(len(n))]

for i in range(total):
    mu_container.append(np.random.normal(i - total / 2, 0.5))
    std_container.append(np.random.normal(0, 0.6) ** 2 + 1e-5)

mu_container.sort()

for i in range(len(n)):
    temp = np.random.choice([_ for _ in range(total)],
                            ncluster[i], replace=False)
    for j in temp:
        mu[i].append(mu_container[j])
        std[i].append(std_container[j])
        uniqueIdx.add(j)

for i in range(len(n)):
    length = 0
    while length < n[i]:
        subLen = min(np.random.poisson(
            np.random.exponential(4)) + 1, n[i] - length)
        length += subLen
        idx = np.random.choice([_ for _ in range(ncluster[i])])
        # data[i].append(np.random.normal(mu[i][idx], 0.5, subLen))
        data[i].append(np.random.normal(mu[i][idx], std[i][idx], subLen))

data = [np.concatenate(data[i]) for i in range(len(n))]

summarydata = np.concatenate(data)

stdvar = np.std(summarydata)
mean = np.mean(summarydata)
for i in range(len(n)):
    data[i] = (data[i] - mean) / stdvar

idx2cluster, dp_ncluster = Gibbs(
    data=data, k=0.005, beta_0=0.0001, dp_1=0.007, dp_2=100.0, niterate=niterate)

for i in range(len(n)):
    data[i] = data[i] * stdvar + mean

summaryidx = []
summarydataidx = []
summaryclusteridx = []
for i in range(len(n)):
    summaryidx.extend([i + 1 for i in range(len(data[i]))])
    summarydataidx.extend([i] * len(data[i]))
    summaryclusteridx.extend(idx2cluster[i])

plt.figure(figsize=(16, 8))
df = pd.DataFrame([[summaryidx[i], summarydata[i], summarydataidx[i] + 1, summaryclusteridx[i] + 1]
                   for i in range(len(summarydata))], columns=["index", "y", "dataset", "cluster"])
fig = sns.scatterplot(data=df, x="index", y="y", style="dataset", hue="cluster",
                      palette=sns.color_palette("tab20", dp_ncluster))
fig.legend(loc="best", fontsize="small", handlelength=0.5)
plt.savefig()
plt.close()
