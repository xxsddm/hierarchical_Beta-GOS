import numpy as np
from math import gamma
from bisect import bisect_left
from typing import List

clusterCount = 0
alpha0, beta0, beta0_alpha0 = 1, 0.5, 0.5
dp1 = dp2 = 2.0
gamma0 = 1
mu0 = 1
samplesize = 0


class Cluster:
    k = 1.0

    def __init__(self, x: float) -> None:
        self.inv_var = np.random.gamma(shape=alpha0, scale=1 / beta0)   # prior: E=alpha/beta; var=alpha/(beta * beta)
        # self.var = 1 / self.inv_var
        inv_sigma = np.sqrt(self.inv_var)
        self.log_inv_sigma = np.log(inv_sigma)
        self.mu = np.random.normal(mu0, 1 / np.sqrt(self.inv_var * Cluster.k))  # prior: E=mu; var=k * self.var
        self.size = 1
        self.cumsum = x
        self.squaresum = x * x

    def add(self, x: float) -> None:
        self.size += 1
        self.cumsum += x
        self.squaresum += x * x

    def remove(self, x: float) -> None:
        self.size -= 1
        self.cumsum -= x
        self.squaresum -= x * x

    def lnprob(self, x: float) -> float:
        temp = x - self.mu
        return self.log_inv_sigma - temp * temp * self.inv_var / 2

    def update(self) -> None:
        mun = (Cluster.k * mu0 + self.cumsum) / (Cluster.k + self.size)
        betan = beta0 + (self.squaresum - self.cumsum * self.cumsum / self.size) / 2
        betan += Cluster.k * (self.cumsum * self.cumsum / self.size + mu0 * mu0 * self.size - 2 * mu0 * self.cumsum) \
            / (2 * (Cluster.k + self.size))
        self.inv_var = np.random.gamma(shape=alpha0 + self.size / 2, scale=1 / betan)
        self.log_inv_sigma = 0.5 * np.log(self.inv_var)
        self.mu = np.random.normal(mun, 1 / np.sqrt((Cluster.k + self.size) * self.inv_var))


clusters = dict[int, Cluster]()


def prob_newcluster(x: float) -> float:
    alphan = alpha0 + 0.5
    kn = Cluster.k + 1
    temp0 = x - mu0
    betan = beta0 + Cluster.k * temp0 * temp0 / (2 * kn)
    temp1 = gamma(alphan) * beta0_alpha0 / gamma0
    temp2 = np.sqrt(Cluster.k / kn)
    return temp1 * temp2 / (betan ** alphan)


def relink(data: np.ndarray, idx2cluster: List[int]) -> None:
    global clusterCount
    to_update = set[int]()
    for i in range(len(data)):
        prev = clusters[idx2cluster[i]]
        prev.remove(data[i])
        if prev.size == 0:
            clusters.pop(idx2cluster[i])
        else:
            to_update.add(idx2cluster[i])
    for i in to_update:
        if i in clusters:
            clusters[i].update()
    base = samplesize - len(data) + dp1
    counter = dict[int, int]()
    for i in range(len(data)):
        prob = np.zeros(len(clusters) + 1, dtype=np.float64)
        id2id = dict[int, int]()
        idx = 0
        x = data[i]
        for j in clusters:
            p = np.exp(clusters[j].lnprob(x))
            prob[idx] = clusters[j].size / base * dp2 * p
            if j in counter:
                prob[idx] += counter[j] * p
            id2id[idx] = j
            idx += 1
        prob[idx] = prob_newcluster(x) * dp1 * dp2 / base
        prob = prob / np.mean(prob)
        for j in range(1, len(prob)):
            prob[j] += prob[j - 1]
        nextIdx = bisect_left(prob, np.random.uniform(low=0.0, high=prob[-1]))
        if nextIdx == idx:
            nextIdx = clusterCount
            clusters[clusterCount] = Cluster(x)
            counter[clusterCount] = 1
            clusterCount += 1
        else:
            nextIdx = id2id[nextIdx]
            clusters[nextIdx].add(x)
            if nextIdx not in counter:
                counter[nextIdx] = 0
            counter[nextIdx] += 1
        idx2cluster[i] = nextIdx
        clusters[nextIdx].update()
        base += 1


# 支持cluster方差不共享
def Gibbs(data: List[np.ndarray], mu_0=0.0, k=0.005, alpha_0=2, beta_0=0.0063, niterate=100,
          dp_1: float = 2.0, dp_2: float = 2.0) -> (List[List[int]], int):
    global clusterCount, mu0, alpha0, beta0, gamma0, beta0_alpha0, dp1, dp2, samplesize
    mu0 = mu_0
    alpha0, beta0, Cluster.k = alpha_0, beta_0, max(k, 0.001)
    gamma0, beta0_alpha0 = gamma(alpha0), beta0 ** alpha0
    dp1, dp2 = dp_1, dp_2
    ndata = len(data)
    clusterCount = 0
    samplesize = 0
    ndrop = max(50, int(0.3 * niterate + 0.0001))
    idx2cluster = [[-1] * len(data[i]) for i in range(ndata)]
    for i in range(ndata):
        samplesize += len(data[i])
        for j in range(len(data[i])):
            idx2cluster[i][j] = clusterCount
            clusters[clusterCount] = Cluster(data[i][j])
            clusterCount += 1
    for i in clusters:
        clusters[i].update()
    llf_square = llf_sum = 0.0
    mu_sum = [np.zeros(len(data[i]), dtype=np.float) for i in range(len(data))]
    inv_var_sum = [np.zeros(len(data[i]), dtype=np.float) for i in range(len(data))]
    for t in range(niterate):
        for i in range(ndata):
            relink(data[i], idx2cluster[i])
        print("{:d} iteration: {:d}".format(t + 1, len(clusters)))
        if t >= ndrop:
            llf = 0.0
            for i in range(ndata):
                for j in range(len(data[i])):
                    cluster = clusters[idx2cluster[i][j]]
                    temp1 = cluster.mu
                    temp2 = cluster.inv_var
                    llf += temp2 / 2 * ((data[i][j] - temp1) ** 2) + np.log(temp2) / 2
                    mu_sum[i][j] += temp1
                    inv_var_sum[i][j] += temp2
            llf_square += llf * llf
            llf_sum += llf
    nrest = niterate - ndrop
    if nrest > 0:
        for i in range(ndata):
            mu_sum[i] /= nrest
            inv_var_sum[i] /= nrest
        dic = 2 * sum([sum([(data[i][j] - mu_sum[i][j]) ** 2 / 2 * inv_var_sum[i][j]
                            - np.log(inv_var_sum[i][j]) / 2
                            for j in range(len(data[i]))])
                       for i in range(ndata)]) \
            + llf_square / nrest - (llf_sum / nrest) ** 2
        print("DIC: {:.4f}".format(dic))
    id2id = dict[int, int]()
    container = [(i, clusters[i].mu) for i in clusters]
    container.sort(key=lambda x: x[1])
    for i in range(len(container)):
        id2id[container[i][0]] = i
    for i in range(ndata):
        for j in range(len(data[i])):
            idx2cluster[i][j] = id2id[idx2cluster[i][j]]
    return idx2cluster, len(clusters)
