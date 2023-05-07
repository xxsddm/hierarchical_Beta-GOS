import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Gibbs import Gibbs, Info

np.random.seed(4399)

n = [200, 175, 200, 150, 225, 250]
niterate = 100
total = 15
ncluster = [min(np.random.poisson(5) + 3, total - 1) for i in range(len(n))]
mu_container = []
std_container = []
mu = [[] for _ in range(len(n))]
std = [[] for _ in range(len(n))]
uniqueIdx = set()
data = [[] for _ in range(len(n))]
title = "ai=3, bi=1"

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

real_info = []
for i in uniqueIdx:
    real_info.append(Info(mu=mu_container[i], sigma=std_container[i]))
sorted(real_info, key=lambda x: x.mu)

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

an = [np.array([3 for _ in range(n[i])]) for i in range(len(n))]
bn = [np.array([1 for _ in range(n[i])]) for i in range(len(n))]
c = [np.array([j for j in range(len(data[i]))], dtype=int)
     for i in range(len(n))]
summarydata = np.concatenate(data)

stdvar = np.std(summarydata)
mean = np.mean(summarydata)
for i in range(len(n)):
    data[i] = (data[i] - mean) / stdvar

info_hat, idx2cluster = Gibbs(data=data, an=an, bn=bn, c=c,
                              dp=0.005, niterate=niterate, draw=True,
                              title=title)

stat_counter = [0] * len(info_hat)
stat_mu = np.zeros(len(info_hat), dtype=np.float64)
stat_var = np.zeros(len(info_hat), dtype=np.float64)

for info in info_hat:
    info.mu = info.mu * stdvar + mean
    info.sigma *= stdvar

for i in range(len(n)):
    data[i] = data[i] * stdvar + mean
    for j in range(len(data[i])):
        stat_counter[idx2cluster[i][j]] += 1
        stat_mu[idx2cluster[i][j]] += data[i][j]
        stat_var[idx2cluster[i][j]] += data[i][j] * data[i][j]

for i in range(len(info_hat)):
    if stat_counter[i] == 1:
        stat_var[i] = 0.0
    else:
        stat_var[i] /= stat_counter[i] - 1
        stat_var[i] -= stat_mu[i] / stat_counter[i] * \
            stat_mu[i] / (stat_counter[i] - 1)
    stat_mu[i] /= stat_counter[i]

print("实际均值: ", [float("{:.4f}".format(real_info[i].mu))
      for i in range(len(real_info))])
print("估计均值: ", [float("{:.4f}".format(stat_mu[i]))
      for i in range(len(info_hat))])
# print("估计均值: ", [float("{:.4f}".format(info_hat[i].mu))
#       for i in range(len(info_hat))])
print("实际标准差: ", [float("{:.4f}".format(real_info[i].sigma))
      for i in range(len(real_info))])
print("估计标准差: ", [float("{:.4f}".format(np.sqrt(stat_var[i])))
      for i in range(len(info_hat))])
# print("估计标准差: ", [float("{:.4f}".format(info_hat[i].sigma))
#       for i in range(len(info_hat))])

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
                      palette=sns.color_palette("tab20", len(info_hat)))
fig.legend(loc="best", fontsize="small", handlelength=0.5)
plt.savefig("data/summary.png")
plt.close()

plt.figure(figsize=(20, 16))
for i in range(len(n)):
    plt.subplot(321 + i)
    plt.title("dataset {:d}".format(i + 1))
    temp = pd.DataFrame([[j + 1, data[i][j], idx2cluster[i][j] + 1] for j in range(len(data[i]))],
                        columns=["index", "y", "cluster"])
    sns.scatterplot(data=temp, x="index", y="y", hue="cluster", legend=False,
                    palette=sns.color_palette("tab20", len(set(idx2cluster[i]))))
plt.savefig("data/subplot.png")
plt.close()
