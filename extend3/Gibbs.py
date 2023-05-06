import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from math import gamma
from bisect import bisect_left
from typing import List

clusterCount = 0
alpha0, beta0, beta0_alpha0 = 1, 0.5, 0.5
gamma0 = 1
mu0 = 0
samplesize = 0
p = np.zeros(3, dtype=np.float64)
lnp = np.zeros(3, dtype=np.float64)


class Info:
    def __init__(self, mu: float = 0.0, sigma: float = 1.0, size: int = 1, idx: int = -1):
        self.mu = mu
        self.sigma = sigma
        self.size = size
        self.idx = idx


class Cluster:
    k = 1

    def __init__(self, cumsum: float, squaresum: float, size: int = 1) -> None:
        self.inv_var = np.random.gamma(shape=alpha0, scale=1 / beta0)  # prior: E=alpha/beta; var=alpha/(beta * beta)
        # self.var = 1 / self.inv_var
        self.inv_sigma = np.sqrt(self.inv_var)
        self.mu = np.random.normal(mu0, 1 / self.inv_sigma / np.sqrt(Cluster.k))  # prior: E=mu; var=k * self.var
        self.size = size
        self.cumsum = cumsum
        self.squaresum = squaresum

    def add(self, cumsum: float, squaresum: float, size: int) -> None:
        self.size += size
        self.cumsum += cumsum
        self.squaresum += squaresum
        # if self.squaresum < 0:
        #     print("add<0:", self.squaresum)

    def remove(self, cumsum: float, squaresum: float, size: int) -> None:
        self.size -= size
        self.cumsum -= cumsum
        self.squaresum -= squaresum
        # if self.squaresum < 0:
        #     print("remove<0:", self.squaresum)

    def prob(self, cumsum: float, squaresum: float, size: int) -> float:
        return math.exp((2 * self.mu * cumsum - size * self.mu * self.mu - squaresum) * self.inv_var / 2) \
            * math.pow(self.inv_sigma, size)

    def lnprob(self, cumsum: float, squaresum: float, size: int) -> float:
        return (2 * self.mu * cumsum - size * self.mu * self.mu - squaresum) * self.inv_var / 2 \
            + size * math.log(self.inv_sigma)

    def update(self) -> None:
        mun = (Cluster.k * mu0 + self.cumsum) / (Cluster.k + self.size)
        betan = beta0 + (self.squaresum - self.cumsum * self.cumsum / self.size) / 2
        betan += Cluster.k * (self.cumsum * self.cumsum / self.size + mu0 * mu0 * self.size - 2 * mu0 * self.cumsum) \
            / (2 * (Cluster.k + self.size))
        self.inv_var = np.random.gamma(shape=alpha0 + self.size / 2, scale=1 / betan)
        self.inv_sigma = math.sqrt(self.inv_var)
        self.mu = np.random.normal(mun, 1 / self.inv_sigma / math.sqrt(Cluster.k + self.size))


clusters = dict[int, Cluster]()


def prob_newcluster(cumsum: float, squaresum: float, size: int) -> float:
    alphan = alpha0 + size / 2
    kn = Cluster.k + size
    betan = beta0 + (squaresum - cumsum * cumsum / size) / 2
    betan += Cluster.k * (cumsum * cumsum / size + mu0 * mu0 * size - 2 * mu0 * cumsum) / (2 * kn)
    temp = gamma(alphan) * beta0_alpha0 / gamma0 * math.pow(Cluster.k / kn, 0.5)
    return temp / math.pow(betan, alphan)


def lnprob_newcluster(cumsum: float, squaresum: float, size: int) -> float:
    alphan = alpha0 + size / 2
    kn = Cluster.k + size
    betan = beta0 + (squaresum - cumsum * cumsum / size) / 2
    betan += Cluster.k * (cumsum * cumsum / size + mu0 * mu0 * size - 2 * mu0 * cumsum) / (2 * kn)
    temp1 = math.log(gamma(alphan) * beta0_alpha0 / gamma0)
    temp2 = math.log(Cluster.k / kn) / 2
    return temp1 - alphan * math.log(betan) + temp2


def GibbsC(data: np.ndarray, w: np.ndarray, c: np.ndarray, idx2cluster: List[int], dp: float) -> None:
    global clusterCount
    length = len(data)
    lnw = np.log(w)
    lnweight = np.log(1 - w)
    size = np.ones(length, dtype=np.int32)
    cumsum = data.copy()
    squaresum = data * data
    cumlnw = 0.0
    base = samplesize - length + dp
    for i in range(length - 1, -1, -1):
        if c[i] == i:
            prev = clusters[idx2cluster[i]]
            if prev.size == size[i]:
                clusters.pop(idx2cluster[i])
            else:
                prev.remove(cumsum[i], squaresum[i], size[i])
                clusters[idx2cluster[i]].update()
        else:
            size[c[i]] += size[i]
            cumsum[c[i]] += cumsum[i]
            squaresum[c[i]] += squaresum[i]
    for i in range(length):
        if c[i] != i:
            prev = clusters[idx2cluster[c[i]]]
            prev.remove(cumsum[i], squaresum[i], size[i])
        exlength = i + len(clusters) + 1
        # # 检验权重和为1
        # checksum1 = np.sum(np.exp(lnweight[: i])) + np.exp(cumlnw)
        # if abs(checksum1 - 1) > 1e-6:
        #     print("检验权重和为1: ", checksum1)
        for prevnode in range(i):
            lnp[prevnode] = lnweight[prevnode] + \
                clusters[idx2cluster[prevnode]].lnprob(cumsum[i], squaresum[i], size[i])
            lnweight[prevnode] += lnw[i]
        k = 0
        idx2key = dict[int, int]()
        for clusterIdx in clusters:
            cluster = clusters[clusterIdx]
            lnp[i + k] = math.log(cluster.size / base) + cumlnw + cluster.lnprob(cumsum[i], squaresum[i], size[i])
            idx2key[i + k] = clusterIdx
            k += 1
        lnp[i + k] = math.log(dp / base) + cumlnw + lnprob_newcluster(cumsum[i], squaresum[i], size[i])
        maxlnp = -1e9
        for j in range(exlength):
            maxlnp = max(maxlnp, lnp[j])
        for j in range(exlength):
            p[j] = math.exp(lnp[j] - maxlnp)
            if j:
                p[j] += p[j - 1]
        prev = bisect_left(p, np.random.uniform(low=0.0, high=p[exlength - 1]), hi=exlength)
        cumlnw += lnw[i]
        base += 1
        if prev < i:
            c[i] = prev
            idx2cluster[i] = idx2cluster[prev]
            clusters[idx2cluster[i]].add(cumsum[i], squaresum[i], size[i])
            clusters[idx2cluster[i]].update()
        elif prev == exlength - 1:
            c[i] = i
            idx2cluster[i] = clusterCount
            temp = Cluster(cumsum[i], squaresum[i], size[i])
            temp.update()
            clusters[clusterCount] = temp
            clusterCount += 1
        else:
            c[i] = i
            idx2cluster[i] = idx2key[prev]
            clusters[idx2cluster[i]].add(cumsum[i], squaresum[i], size[i])
            clusters[idx2cluster[i]].update()


def GibbsW(w: np.ndarray, c: np.ndarray, an: np.ndarray, bn: np.ndarray) -> None:
    length = len(c)
    A, B = [0] * (length + 1), [0] * length
    for i in range(length - 1, -1, -1):
        if i:
            A[i - 1] += 1
        if c[i] != i:
            A[c[i]] -= 1
            B[c[i]] += 1
        A[i] += A[i + 1]
        w[i] = np.random.beta(an[i] + A[i], bn[i] + B[i])


# 支持cluster方差不共享
def Gibbs(data: List[np.ndarray], an: List[np.ndarray], bn: List[np.ndarray],
          mu_0=0.0, k=0.005, alpha_0=2, beta_0=0.0063, niterate=100, dp: float = 2.0,
          c: List[np.ndarray] = None, draw: bool = False, title: str = "", path: str = None) \
            -> (List[Info], List[List[int]]):
    global clusterCount, mu0, alpha0, beta0, gamma0, beta0_alpha0, samplesize, p, lnp
    mu0 = mu_0
    alpha0, beta0, Cluster.k = alpha_0, beta_0, max(k, 0.001)
    gamma0, beta0_alpha0 = gamma(alpha0), beta0 ** alpha0
    ndata = len(data)
    w = [np.random.beta(an[i], bn[i]) for i in range(ndata)]
    num_cluster = []
    if c is None:
        c = [np.array([j for j in range(len(data[i]))], dtype=np.int32) for i in range(ndata)]
    elif c[0].__class__ == list:
        c = [np.array([j for j in range(len(data[i]))], dtype=np.int32) for i in range(ndata)]
        # c = [np.array(c[i], dtype=np.int32) for i in range(ndata)]
    idx2cluster = [[-1] * len(data[i]) for i in range(ndata)]
    samplesize = 0
    llf_square = llf_sum = 0.0
    ndrop = max(50, int(0.3 * niterate + 0.0001))
    for i in range(ndata):
        samplesize += len(data[i])
        for j in range(len(data[i])):
            idx2cluster[i][j] = clusterCount
            clusters[clusterCount] = Cluster(cumsum=data[i][j], squaresum=data[i][j] * data[i][j])
            clusters[clusterCount].update()
            clusterCount += 1
    mu_sum = [np.zeros(len(data[i]), dtype=np.float) for i in range(len(data))]
    inv_var_sum = [np.zeros(len(data[i]), dtype=np.float) for i in range(len(data))]
    p = np.zeros(samplesize + 5, dtype=np.float64)
    lnp = np.zeros(samplesize + 5, dtype=np.float64)
    for i in range(niterate):
        for j in range(ndata):
            GibbsC(data[j], w[j], c[j], idx2cluster[j], dp)
            GibbsW(w[j], c[j], an[j], bn[j])
        num_cluster.append(len(clusters))
        print("{:d} iteration: {:d}".format(i + 1, len(clusters)))
        if i >= ndrop:
            llf = 0.0
            for j in range(ndata):
                for k in range(len(data[j])):
                    cluster = clusters[idx2cluster[j][k]]
                    temp1 = cluster.mu
                    temp2 = cluster.inv_var
                    llf += temp2 / 2 * ((data[j][k] - temp1) ** 2) + np.log(temp2) / 2
                    mu_sum[j][k] += temp1
                    inv_var_sum[j][k] += temp2
    nrest = niterate - ndrop
    dic = 1e9
    # prob = None
    if nrest > 0:
        for i in range(ndata):
            mu_sum[i] /= nrest
            inv_var_sum[i] /= nrest
        dic = 2 * sum([sum([(data[i][j] - mu_sum[i][j]) ** 2 / 2 * inv_var_sum[i][j]
                            - np.log(inv_var_sum[i][j]) / 2
                            for j in range(len(data[i]))])
                       for i in range(ndata)]) \
            + llf_square / nrest - (llf_sum / nrest) ** 2
        # prob = [np.array([np.exp((data[i][j] - clusters[idx2cluster[i][j]].mu) ** 2
        #                          * (clusters[idx2cluster[i][j]].inv_var / 2)) *
        #                   clusters[idx2cluster[i][j]].inv_sigma
        #                   for j in range(len(data[i]))]) for i in range(ndata)]
        print("DIC: {:.4f}".format(dic))
    if draw:
        plt.figure(figsize=(15, 8))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        if title:
            plt.title("分配过程(" + title + ")  (dp={dp}, k={k}, alpha0={alpha_0}, beta0={beta_0}, DIC={dic:.2f})"
                      .format(dp=dp, k=Cluster.k, alpha_0=alpha_0, beta_0=beta_0, dic=dic))
        else:
            plt.title("分配过程  (dp={dp}, k={k}, alpha0={alpha_0}, beta0={beta_0}, DIC={dic})"
                      .format(dp=dp, k=Cluster.k, alpha_0=alpha_0, beta_0=beta_0, dic=dic))
        plt.plot([i + 1 for i in range(niterate)], num_cluster, 'bo-', markersize=2)
        plt.xlabel("迭代次数")
        plt.ylabel("剩余类别数")
        plt.savefig(path)
    # cluster序号按均值排序和标号
    # 重新编排样本对应cluster
    info = [Info() for _ in range(len(clusters))]
    id2id = dict[int, int]()
    idx = 0
    for i in clusters:
        cluster = clusters[i]
        info[idx].mu = cluster.mu
        info[idx].sigma = 1.0 / cluster.inv_sigma
        info[idx].size = cluster.size
        info[idx].idx = idx
        id2id[i] = idx
        idx += 1
    for i in range(ndata):
        for j in range(len(data[i])):
            idx2cluster[i][j] = id2id[idx2cluster[i][j]]
    id2id.clear()
    info.sort(key=lambda x: x.mu)
    for i in range(len(info)):
        id2id[info[i].idx] = i
        info[i].idx = i
    for i in range(ndata):
        for j in range(len(data[i])):
            idx2cluster[i][j] = id2id[idx2cluster[i][j]]
    return info, idx2cluster
