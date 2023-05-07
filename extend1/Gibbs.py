import math
import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect_left
from scipy.stats import invgamma

clusterCount = 0
mu0 = sigma0 = 1
var0 = var = 1
sigma = 1
totalblock = 0
samplesize = 0
lazy_size = np.zeros(3, dtype=np.int32)
lazy_cumsum = np.zeros(3, dtype=np.float64)
lazy_squaresum = np.zeros(3, dtype=np.float64)
p = np.zeros(3, dtype=np.float64)
lnp = np.zeros(3, dtype=np.float64)


class Cluster:
    def __init__(self, cumsum: float, squaresum: float, size: int = 1, mu: float = 0) -> None:
        self.mu = mu
        self.size = size
        self.count = 1
        self.cumsum = cumsum
        self.squaresum = squaresum

    def add(self, cumsum: float, squaresum: float, size: int, block: bool = False) -> None:
        self.size += size
        self.cumsum += cumsum
        self.squaresum += squaresum
        if block:
            self.count += 1
        # if self.squaresum < 0:
        #     print("add<0:", self.squaresum)

    def remove(self, cumsum: float, squaresum: float, size: int, block: bool = False) -> None:
        self.size -= size
        self.cumsum -= cumsum
        self.squaresum -= squaresum
        if block:
            self.count -= 1
        # if self.squaresum < 0:
        #     print("remove<0:", self.squaresum)

    def lnprob(self, cumsum: float, size: int) -> float:
        return (2 * self.mu * cumsum - size * self.mu * self.mu) / (2 * var)

    def update(self) -> None:
        mu = (self.cumsum * var0 + mu0 * var) / \
             (self.size * var0 + var)
        std_var = sigma0 * sigma / np.sqrt(self.size * var0 + var)
        self.mu = np.random.normal(mu, std_var)


clusters = dict[int, Cluster]()


def lnprobnew(cumsum: float, count: int) -> float:
    temp1 = sigma / math.sqrt(count * var0 + var)
    temp2 = var0 * cumsum * cumsum \
            + 2 * mu0 * var * cumsum \
            - count * var * mu0 * mu0
    temp3 = 2 * var * (count * var0 + var)
    return math.log(temp1) + temp2 / temp3


def GibbsC(data: np.ndarray, w: np.ndarray, c: np.ndarray, idx2cluster: list[int],
           size: np.ndarray, cumsum: np.ndarray, squaresum: np.ndarray, dp: float) -> None:
    global clusterCount, totalblock
    global lazy_size, lazy_cumsum, lazy_squaresum
    length = len(data)
    # size = np.ones_like(data, dtype=np.int32)
    # cumsum = data.copy()
    # squaresum = data * data
    # for i in range(length - 1, -1, -1):
    #     if c[i] == i:
    #         continue
    #     size[c[i]] += size[i]
    #     cumsum[c[i]] += cumsum[i]
    #     squaresum[c[i]] += squaresum[i]
    lnw = np.log(w)
    lnweight = np.log(1 - w)
    cumlnw = 0.0
    for i in range(length - 1, -1, -1):
        lnweight[i] += cumlnw
        cumlnw += lnw[i]
    for i in range(length - 1, -1, -1):
        cumlnw -= lnw[i]
        lnweight[i] = cumlnw
        size[i] += lazy_size[i]
        cumsum[i] += lazy_cumsum[i]
        squaresum[i] += lazy_squaresum[i]
        if c[i] == i:
            totalblock -= 1
            if clusters[idx2cluster[i]].count == 1:
                clusters.pop(idx2cluster[i])
            else:
                clusters[idx2cluster[i]].remove(cumsum[i], squaresum[i], size[i], True)
                # clusters[idx2cluster[i]].update()
        else:
            j = c[i]
            lazy_size[j] += lazy_size[i]
            lazy_cumsum[j] += lazy_cumsum[i]
            lazy_squaresum[j] += lazy_squaresum[i]
            lazy_size[j] -= size[i]
            lazy_cumsum[j] -= cumsum[i]
            lazy_squaresum[j] -= squaresum[i]
            clusters[idx2cluster[i]].remove(cumsum[i], squaresum[i], size[i])
            # clusters[idx2cluster[i]].update()
        lazy_size[i] = 0
        lazy_cumsum[i] = 0.0
        lazy_squaresum[i] = 0.0
        idx2cluster[i] = -1
        exlength = i + len(clusters) + 1
        lnp = np.zeros(exlength, dtype=np.float)
        for prevnode in range(i):
            lnweight[prevnode] -= lnw[i]
            lnp[prevnode] = lnweight[prevnode] + \
                            clusters[idx2cluster[prevnode]].lnprob(cumsum[i], size[i])
        # 检验权重和为1
        # checksum1 = np.sum(np.exp(lnweight[: (i + 1)]))
        # if abs(checksum1 - 1) > 1e-6:
        #     print("检验权重和为1: ", checksum1)
        k = 0
        idx2key = dict()
        for clusterIdx in clusters:
            cluster = clusters[clusterIdx]
            lnp[i + k] = math.log(cluster.count / (totalblock + dp)) + \
                         lnweight[i] + cluster.lnprob(cumsum[i], size[i])
            idx2key[i + k] = clusterIdx
            k += 1
        lnp[i + k] = np.log(dp / (totalblock + dp)) \
                     + lnweight[i] + lnprobnew(cumsum[i], size[i])
        p = np.cumsum(np.exp(lnp - np.max(lnp)))
        prev = bisect_left(p, np.random.uniform(low=0.0, high=p[-1]))
        if prev < i:
            c[i] = prev
            j = c[i]
            lazy_size[j] += size[i]
            lazy_cumsum[j] += cumsum[i]
            lazy_squaresum[j] += squaresum[i]
            idx2cluster[i] = idx2cluster[j]
            clusters[idx2cluster[i]].add(cumsum[i], squaresum[i], size[i])
            # clusters[idx2cluster[i]].update()
        elif prev == exlength - 1:
            c[i] = i
            idx2cluster[i] = clusterCount
            temp = Cluster(cumsum[i], squaresum[i], size[i])
            temp.update()
            clusters[clusterCount] = temp
            clusterCount += 1
            totalblock += 1
        else:
            c[i] = i
            idx2cluster[i] = idx2key[prev]
            clusters[idx2cluster[i]].add(cumsum[i], squaresum[i], size[i], True)
            # clusters[idx2cluster[i]].update()
            totalblock += 1
    for i in range(1, length):
        idx2cluster[i] = idx2cluster[c[i]]
    # 检验连通块数
    # a = 0
    # for b in clusters:
    #     a += clusters[b].count
    # if a != totalblock:
    #     print("检验连通块数量: ", a == totalblock)


def GibbsParameter(alpha0: float, beta0: float) -> float:
    temp = 0
    for idx in clusters:
        cluster = clusters[idx]
        # P16整体方差
        temp1 = cluster.size / 2
        temp2 = (cluster.squaresum + cluster.mu * cluster.mu * cluster.size) / 2 - cluster.mu * cluster.cumsum
        # if beta0 + temp2 < 0:
        #     print("std < 0: ", beta0 + temp2)
        #     print(temp1, temp2, cluster.mu, cluster.cumsum, cluster.squaresum)
        temp += invgamma.rvs(alpha0 + temp1, scale=beta0 + temp2) * (cluster.size - 1)
    return temp / (samplesize - len(clusters))


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


# cluster共享方差
def Gibbs(data: list[np.ndarray], an: list[np.ndarray], bn: list[np.ndarray],
          mu_0=0, sigma_0=0.5, alpha0=2.004, beta0=0.0063, niterate=100, dp: float = 2.0,
          c: list[np.ndarray] = None, draw: bool = False, title: str = "", path: str = None) \
        -> tuple[list[float], list[list[int]]]:
    global clusterCount, mu0, sigma0, var0, sigma, var, totalblock, samplesize
    global lazy_size, lazy_cumsum, lazy_squaresum
    mu0, sigma0 = mu_0, sigma_0
    var0 = sigma0 * sigma0
    ndata = len(data)
    var = invgamma.rvs(alpha0, scale=beta0)  # P16整体方差
    sigma = math.sqrt(var)
    w = [np.random.beta(an[i], bn[i]) for i in range(ndata)]
    size = [np.ones(len(data[i]), dtype=np.int32) for i in range(ndata)]
    cumsum = [data[i].copy() for i in range(ndata)]
    squaresum = [(data[i] * data[i]) for i in range(ndata)]
    num_cluster = []
    max_size = max(len(data[i]) for i in range(ndata))
    lazy_size = np.zeros(max_size, dtype=np.int32)
    lazy_cumsum = np.zeros(max_size, dtype=np.float64)
    lazy_squaresum = np.zeros(max_size, dtype=np.float64)
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
            clusters[clusterCount] = Cluster(cumsum=cumsum[i][j], squaresum=squaresum[i][j])
            clusters[clusterCount].update()
            clusterCount += 1
    totalblock = clusterCount
    means = np.zeros(samplesize, dtype=np.float)
    mu_sum = [np.zeros(len(data[i]), dtype=np.float) for i in range(len(data))]
    var_sum = 0.0
    for i in range(niterate):
        for j in range(ndata):
            GibbsC(data[j], w[j], c[j], idx2cluster[j], size[j], cumsum[j], squaresum[j], dp)
            GibbsW(w[j], c[j], an[j], bn[j])
        idx = 0
        for j in clusters:
            clusters[j].update()
            means[idx] = clusters[j].mu
            idx += 1
        num_cluster.append(len(clusters))
        var = GibbsParameter(alpha0, beta0)
        sigma = np.sqrt(var)
        print("{:d} iteration: {:d}".format(i + 1, len(clusters)))
        if i >= ndrop:
            llf = 0.0
            var_sum += var
            for j in range(ndata):
                for k in range(len(data[j])):
                    temp = clusters[idx2cluster[j][k]].mu
                    llf += (data[j][k] - temp) ** 2 / (2 * var)
                    mu_sum[j][k] += temp
            llf_square += llf * llf
            llf_sum += llf
    nrest = niterate - ndrop
    var_sum /= nrest
    dic = 2 * sum([len(data[i]) for i in range(ndata)]) * np.log(sigma) - \
          2 * sum([sum([(data[i][j] - (mu_sum[i][j] / nrest)) ** 2 / (2 * var_sum)
                        for j in range(len(data[i]))])
                   for i in range(ndata)]) \
          + llf_square / nrest - (llf_sum / nrest) ** 2
    print("DIC: {:.4f}".format(dic))
    if draw:
        plt.figure(figsize=(15, 8))
        if title:
            plt.title("分配过程(" + title + ") DIC={:.4f}".format(dic))
        else:
            plt.title("分配过程 DIC={:.4f}".format(dic))
        plt.plot([i + 1 for i in range(niterate)], num_cluster, 'bo-', markersize=2)
        plt.xlabel("分配轮数")
        plt.ylabel("剩余类别数")
        if path:
            plt.savefig(path)
        else:
            plt.savefig("path of n_cluster.png")
    return means[: len(clusters)], idx2cluster
